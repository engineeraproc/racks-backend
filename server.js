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

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '50mb' }));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pc.index(process.env.PINECONE_INDEX);

// Permite receber os ficheiros enviados pelo seu Painel de Administrador
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

// ==========================================
// FUNÇÕES AUXILIARES
// ==========================================
async function getEmbedding(text) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${process.env.GEMINI_API_KEY}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: "models/text-embedding-004", content: { parts: [{ text: text }] } })
    });
    const data = await response.json();
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
// ROTA 1: RECEBER FICHEIROS DO ADMINISTRADOR (UPLOAD)
// ==========================================
app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) throw new Error("Nenhum ficheiro recebido.");
        
        const file = req.file;
        const ext = path.extname(file.originalname).toLowerCase();
        
        // Guarda temporariamente no Render só para processar e apaga a seguir (Poupa memória)
        const tempFilePath = path.join(os.tmpdir(), `${Date.now()}_${file.originalname}`);
        fs.writeFileSync(tempFilePath, file.buffer);

        console.log(`📥 A processar ficheiro: ${file.originalname}`);

        // SE FOR PDF OU EXCEL (Vai para a Google)
        if (ext === '.pdf' || ['.xlsx', '.xls', '.csv'].includes(ext)) {
            let fileToUpload = tempFilePath;
            let mimeType = 'application/pdf';

            // Converte Excel para CSV para a IA ler melhor
            if (['.xlsx', '.xls', '.csv'].includes(ext)) {
                const workbook = xlsx.readFile(tempFilePath);
                const csvData = xlsx.utils.sheet_to_csv(workbook.Sheets[workbook.SheetNames[0]]);
                fileToUpload = path.join(os.tmpdir(), `temp_csv_${Date.now()}.csv`);
                fs.writeFileSync(fileToUpload, csvData);
                mimeType = 'text/csv';
            }

            // Envia para o cérebro da Google
            const uploadResult = await fileManager.uploadFile(fileToUpload, { 
                mimeType: mimeType, 
                displayName: file.originalname 
            });
            
            console.log(`✅ Guardado na Nuvem Google: ${file.originalname}`);
            res.json({ method: 'gemini_file', uri: uploadResult.file.uri, mimeType: mimeType });
        } 
        // SE FOR TEXTO (Vai para o Pinecone)
        else if (ext === '.txt') {
            const textContent = file.buffer.toString('utf-8');
            const chunks = chunkText(textContent, 1000);
            const finalRecords = [];
            
            for (let i = 0; i < chunks.length; i++) {
                if (!chunks[i].trim()) continue;
                const values = await getEmbedding(chunks[i]);
                finalRecords.push({ 
                    id: `${Date.now()}_${i}`, 
                    values: values, 
                    metadata: { text: chunks[i], source: file.originalname } 
                });
            }
            if (finalRecords.length > 0) await pineconeIndex.upsert(finalRecords);
            
            console.log(`📚 Guardado no Pinecone: ${file.originalname}`);
            res.json({ method: 'pinecone' });
        } else {
            throw new Error("Formato não suportado");
        }
    } catch (error) {
        console.error("❌ Erro no upload:", error);
        res.status(500).json({ error: error.message });
    }
});

// ==========================================
// ROTA 2: CHAT INTELIGENTE (PROCURA TUDO NA NUVEM)
// ==========================================
app.post('/api/chat', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        const { query } = req.body;
        console.log(`\n🧠 Pergunta do Utilizador: "${query}"`);
        
        // 1. Pesquisa Pinecone (Manuais em Texto)
        let queryVectorRaw = await getEmbedding(query);
        let queryVector = [];
        for(let i=0; i<queryVectorRaw.length; i++) queryVector.push(Number(queryVectorRaw[i]));
        
        let searchResults = { matches: [] };
        try { searchResults = await pineconeIndex.query({ vector: queryVector, topK: 3, includeMetadata: true }); } catch (e) {}
        let contextText = searchResults.matches ? searchResults.matches.map(m => `[Documento: ${m.metadata?.source}]\n${m.metadata?.text}`).join('\n\n') : '';

        // 2. O TRUQUE MÁGICO: Vai à Google ver TODOS os PDFs que você já enviou!
        let allGeminiFiles = [];
        try {
            const listResult = await fileManager.listFiles();
            if (listResult.files) {
                allGeminiFiles = listResult.files.map(f => ({ uri: f.uri, mimeType: f.mimeType }));
                console.log(`📎 Encontrados ${allGeminiFiles.length} documentos da APROC na nuvem.`);
            }
        } catch(e) { console.error("Aviso: Erro ao listar ficheiros da Google", e); }

        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const systemPrompt = `Você é o RACKS IA, um assistente técnico especializado da APROC.
Responda de forma rigorosa com base nos arquivos anexados a esta conversa e na Base de Conhecimento Rápida abaixo:
${contextText || "Sem contexto rápido."}`;

        const chatParts = [{ text: systemPrompt }, { text: `PERGUNTA: ${query}` }];

        // Anexa todos os PDFs globais na cabeça da IA
        allGeminiFiles.forEach(file => {
            if (file.mimeType && file.uri) chatParts.push({ fileData: { mimeType: file.mimeType, fileUri: file.uri } });
        });

        const resultStream = await model.generateContentStream(chatParts);
        for await (const chunk of resultStream.stream) {
            res.write(`data: ${JSON.stringify({ text: chunk.text() })}\n\n`);
        }
        res.write(`data: [DONE]\n\n`);
        res.end();

    } catch (error) {
        console.error("Erro no Stream de Resposta:", error);
        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
        res.end();
    }
});

app.listen(port, () => {
    console.log(`🚀 SERVIDOR RACKS IA PRONTO E BLINDADO NA PORTA ${port}`);
});