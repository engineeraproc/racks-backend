const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const os = require('os');
const xlsx = require('xlsx');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { GoogleAIFileManager } = require('@google/generative-ai/server');
const { Pinecone } = require('@pinecone-database/pinecone');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pc.index(process.env.PINECONE_INDEX);

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

// ==========================================
// 🤖 INTERROGATÓRIO À GOOGLE (AUTO-DESCOBERTA SEGURA)
// ==========================================
let ACTIVE_EMBEDDING_MODEL = null; 

async function getEmbedding(text) {
    if (!ACTIVE_EMBEDDING_MODEL) {
        console.log("🔍 A interrogar os servidores da Google sobre os modelos autorizados...");
        const urlModels = `https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY}`;
        const resModels = await fetch(urlModels);
        const dataModels = await resModels.json();

        if (!dataModels.models) throw new Error("A Google não devolveu a lista de modelos.");

        const validModels = dataModels.models.filter(m => 
            m.supportedGenerationMethods?.includes("embedContent") && 
            (m.name.includes("text-embedding") || m.name.includes("embedding"))
        );

        if (validModels.length === 0) throw new Error("A sua chave API não possui modelos de texto vetoriais ativos.");

        const model004 = validModels.find(m => m.name.includes("004"));
        ACTIVE_EMBEDDING_MODEL = model004 ? model004.name : validModels[0].name;
        
        console.log(`✅ Sucesso! A Google autorizou o uso exato do modelo: ${ACTIVE_EMBEDDING_MODEL}`);
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/${ACTIVE_EMBEDDING_MODEL}:embedContent?key=${process.env.GEMINI_API_KEY}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: ACTIVE_EMBEDDING_MODEL,
            content: { parts: [{ text: text }] }
        })
    });
    
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error?.message || "Erro na geração do vetor.");
    }
    
    const data = await response.json();
    if (!data.embedding || !Array.isArray(data.embedding.values) || data.embedding.values.length === 0) {
        throw new Error("Formato numérico inválido.");
    }
    return data.embedding.values;
}

function chunkText(text, maxLength = 1000) {
    const chunks = [];
    let currentChunk = '';
    const cleanText = text.replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim();
    const sentences = cleanText.split('. ');

    for (let sentence of sentences) {
        if (currentChunk.length + sentence.length > maxLength) {
            if (currentChunk.trim()) chunks.push(currentChunk.trim());
            currentChunk = sentence + '. ';
        } else {
            currentChunk += sentence + '. ';
        }
    }
    if (currentChunk.trim()) chunks.push(currentChunk.trim());
    return chunks;
}

// ==========================================
// ROTA 1: UPLOAD 
// ==========================================
app.post('/api/upload', upload.single('file'), async (req, res) => {
    let tempFilePath = null;
    try {
        if (!req.file) throw new Error("O ficheiro não chegou ao servidor.");

        const fileName = req.file.originalname;
        const ext = path.extname(fileName).toLowerCase();
        
        console.log(`\n==============================================`);
        console.log(`📥 [UPLOAD] Recebido: ${fileName}`);

        if (ext === '.pdf' || req.file.mimetype === 'application/pdf' || ['.xlsx', '.xls', '.xlsb', '.csv'].includes(ext)) {
            console.log("📄 Enviando para File API (Gemini Nativo)...");
            tempFilePath = path.join(os.tmpdir(), `temp_${Date.now()}${ext || '.pdf'}`);
            if (['.xlsx', '.xls', '.xlsb', '.csv'].includes(ext)) {
                const workbook = xlsx.read(req.file.buffer, { type: 'buffer' });
                const sheetName = workbook.SheetNames[0];
                const csvData = xlsx.utils.sheet_to_csv(workbook.Sheets[sheetName]);
                fs.writeFileSync(tempFilePath, csvData);
                const uploadResult = await fileManager.uploadFile(tempFilePath, { mimeType: 'text/csv', displayName: fileName });
                return res.json({ method: 'gemini_file', uri: uploadResult.file.uri, mimeType: 'text/csv' });
            } else {
                fs.writeFileSync(tempFilePath, req.file.buffer);
                const uploadResult = await fileManager.uploadFile(tempFilePath, { mimeType: 'application/pdf', displayName: fileName });
                return res.json({ method: 'gemini_file', uri: uploadResult.file.uri, mimeType: 'application/pdf' });
            }
        }

        console.log("📝 Enviando texto para o Motor Rápido Pinecone...");
        let textContent = req.file.buffer.toString('utf-8');
        if (textContent.length === 0 || !textContent.trim()) throw new Error("Leitura resultou em 0 caracteres.");

        const chunks = chunkText(textContent, 1000);
        if (chunks.length === 0) throw new Error("A formatação não gerou blocos válidos.");

        const vectorsRaw = [];
        for (let i = 0; i < chunks.length; i++) {
            if (!chunks[i] || chunks[i].trim() === '') continue;
            const values = await getEmbedding(chunks[i]);
            vectorsRaw.push({
                id: `${fileName.replace(/[^a-zA-Z0-9]/g, '_')}-${i}-${Date.now()}`,
                values: values,
                metadata: { text: chunks[i], source: fileName }
            });
        }

        let indexDimension = null;
        try {
            let stats;
            try { stats = await pineconeIndex.describeIndexStats(); } 
            catch(e) { stats = await pineconeIndex.describeIndexStats({ describeIndexStatsRequest: {} }); }
            if (stats && stats.dimension) indexDimension = stats.dimension;
        } catch (e) { }

        const finalRecords = [];
        for (let i = 0; i < vectorsRaw.length; i++) {
            const raw = vectorsRaw[i];
            const safeId = String(raw.id).replace(/[^a-zA-Z0-9_-]/g, '');
            const safeValues = [];
            for (let j = 0; j < raw.values.length; j++) safeValues.push(Number(raw.values[j]));

            if (indexDimension) {
                if (safeValues.length > indexDimension) safeValues.length = indexDimension;
                else while (safeValues.length < indexDimension) safeValues.push(0);
            }

            finalRecords.push({
                id: safeId,
                values: safeValues,
                metadata: { text: String(raw.metadata.text || ""), source: String(raw.metadata.source || "") }
            });
        }

        if (finalRecords.length === 0) throw new Error("Nenhum vetor válido sobreviveu à purificação.");

        let success = false;
        let errorsLogs = [];

        try { await pineconeIndex.upsert(finalRecords); success = true; } catch(e1) { errorsLogs.push(`Array: ${e1.message}`); }
        if (!success) { try { await pineconeIndex.upsert({ records: finalRecords }); success = true; } catch(e2) { errorsLogs.push(`Obj records: ${e2.message}`); } }
        if (!success) { try { await pineconeIndex.upsert({ vectors: finalRecords }); success = true; } catch(e3) { errorsLogs.push(`Obj vectors: ${e3.message}`); } }
        if (!success) { try { await pineconeIndex.upsert({ upsertRequest: { vectors: finalRecords } }); success = true; } catch(e4) { errorsLogs.push(`upsertRequest: ${e4.message}`); } }

        if (!success) throw new Error(`Falha no Pinecone: ${errorsLogs.join(' | ')}`);
        
        return res.json({ method: 'pinecone', message: 'Manual guardado com sucesso!' });

    } catch (error) {
        res.status(500).json({ error: error.message });
    } finally {
        if (tempFilePath && fs.existsSync(tempFilePath)) fs.unlinkSync(tempFilePath);
    }
});

// ==========================================
// ROTA 2: CHAT (AGORA COM TRANSMISSÃO EM TEMPO REAL)
// ==========================================
app.post('/api/chat', async (req, res) => {
    // 🚀 ATIVA O MODO DE "TRANSMISSÃO AO VIVO" (STREAMING)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        const { query, geminiFiles } = req.body;
        if (!query) throw new Error("Pergunta vazia.");

        console.log(`\n🧠 [CHAT STREAMING] Pergunta: "${query}"`);

        let queryVectorRaw = await getEmbedding(query);
        let queryVector = [];
        for(let i=0; i<queryVectorRaw.length; i++) queryVector.push(Number(queryVectorRaw[i]));
        
        let indexDimension = null;
        try {
            let stats;
            try { stats = await pineconeIndex.describeIndexStats(); } 
            catch(e) { stats = await pineconeIndex.describeIndexStats({ describeIndexStatsRequest: {} }); }
            if (stats && stats.dimension) indexDimension = stats.dimension;
        } catch (e) {}

        if (indexDimension) {
            if (queryVector.length > indexDimension) queryVector.length = indexDimension;
            else while (queryVector.length < indexDimension) queryVector.push(0);
        }
        
        let searchResults;
        let querySuccess = false;
        
        try { searchResults = await pineconeIndex.query({ vector: queryVector, topK: 3, includeMetadata: true }); querySuccess = true; } catch (e) {}
        if (!querySuccess) { try { searchResults = await pineconeIndex.query({ queryRequest: { vector: queryVector, topK: 3, includeMetadata: true }}); querySuccess = true; } catch (e) {} }
        if (!querySuccess) { try { searchResults = await pineconeIndex.query({ vectors: [queryVector], topK: 3, includeMetadata: true }); querySuccess = true; } catch (e) {} }
        if (!querySuccess) searchResults = { matches: [] };
        
        let contextText = searchResults.matches ? searchResults.matches.map(m => `[Documento: ${m.metadata?.source || 'Desconhecido'}]\n${m.metadata?.text || ''}`).join('\n\n') : '';

        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const systemPrompt = `Você é o RACKS IA, um assistente técnico especializado da APROC.
Responda de forma rigorosa e EXCLUSIVAMENTE com base nos arquivos anexados a esta conversa e na Base de Conhecimento Rápida abaixo:
${contextText || "Nenhum contexto de texto rápido encontrado."}`;

        const chatParts = [{ text: systemPrompt }, { text: `PERGUNTA DO UTILIZADOR: ${query}` }];

        if (geminiFiles && geminiFiles.length > 0) {
            geminiFiles.forEach(file => {
                if (file.mimeType && file.uri) chatParts.push({ fileData: { mimeType: file.mimeType, fileUri: file.uri } });
            });
        }

        console.log(`📝 A enviar palavras aos bocadinhos para o site...`);
        
        // 🚀 USA A FUNÇÃO DE STREAMING DA GOOGLE
        const resultStream = await model.generateContentStream(chatParts);
        
        // Vai lendo palavra por palavra e enviando para o site imediatamente
        for await (const chunk of resultStream.stream) {
            const chunkText = chunk.text();
            res.write(`data: ${JSON.stringify({ text: chunkText })}\n\n`);
        }
        
        // Finaliza a transmissão
        res.write(`data: [DONE]\n\n`);
        res.end();
        console.log(`✅ Transmissão terminada!`);

    } catch (error) {
        console.error("❌ Erro no chat:", error.message);
        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
        res.end();
    }
});

app.listen(port, () => {
    console.log(`\n===========================================`);
    console.log(`🚀 SERVIDOR RACKS IA (V12 - STREAMING EM TEMPO REAL)`);
    console.log(`📡 Porta: ${port}`);
    console.log(`===========================================\n`);
});