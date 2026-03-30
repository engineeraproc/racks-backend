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

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

// ==========================================
// 🛡️ MEMÓRIA GLOBAL (AUTO-SINCRONIZAÇÃO)
// ==========================================
const GLOBAL_GEMINI_FILES = [];

async function getEmbedding(text) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${process.env.GEMINI_API_KEY}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: "models/text-embedding-004",
            content: { parts: [{ text: text }] }
        })
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

// O MOTOR QUE LÊ A SUA PASTA DO VS CODE
async function syncGlobalDocuments() {
    console.log("\n🔄 Iniciando Sincronização da Pasta 'documentos_globais'...");
    
    const globalDir = path.join(__dirname, 'documentos_globais');
    if (!fs.existsSync(globalDir)) {
        fs.mkdirSync(globalDir);
        console.log("📁 Pasta 'documentos_globais' criada. Coloque os seus PDFs e Excels aqui dentro!");
        return;
    }

    const files = fs.readdirSync(globalDir);
    if (files.length === 0) {
        console.log("📂 A pasta está vazia. Nenhuma diretriz global adicionada.");
        return;
    }

    // Procura arquivos antigos para não gastar o seu limite da Google
    let existingGeminiFiles = [];
    try {
        const listResult = await fileManager.listFiles();
        existingGeminiFiles = listResult.files || [];
    } catch(e) { console.log("Aviso: Falha ao listar arquivos na Google."); }

    for (const file of files) {
        const filePath = path.join(globalDir, file);
        const ext = path.extname(file).toLowerCase();
        const uniqueName = `global_${file}`;

        try {
            // Se for PDF ou Planilha (Vai para o Cérebro Google)
            if (ext === '.pdf' || ['.xlsx', '.xls', '.csv'].includes(ext)) {
                const alreadyUploaded = existingGeminiFiles.find(f => f.displayName === uniqueName);
                
                if (alreadyUploaded) {
                    console.log(`✅ [Memória Ativa] ${file}`);
                    GLOBAL_GEMINI_FILES.push({ uri: alreadyUploaded.uri, mimeType: alreadyUploaded.mimeType });
                } else {
                    console.log(`📤 [Sincronizando] ${file}...`);
                    let fileToUpload = filePath;
                    let mimeType = 'application/pdf';

                    if (['.xlsx', '.xls', '.csv'].includes(ext)) {
                        const workbook = xlsx.readFile(filePath);
                        const csvData = xlsx.utils.sheet_to_csv(workbook.Sheets[workbook.SheetNames[0]]);
                        fileToUpload = path.join(os.tmpdir(), `temp_global_${Date.now()}.csv`);
                        fs.writeFileSync(fileToUpload, csvData);
                        mimeType = 'text/csv';
                    }

                    const uploadResult = await fileManager.uploadFile(fileToUpload, { mimeType: mimeType, displayName: uniqueName });
                    GLOBAL_GEMINI_FILES.push({ uri: uploadResult.file.uri, mimeType: mimeType });
                    console.log(`🚀 [Sucesso] ${file} agora faz parte do cérebro da IA!`);
                }
            } 
            // Se for Texto (Vai para o Pinecone)
            else if (ext === '.txt') {
                const textContent = fs.readFileSync(filePath, 'utf-8');
                const chunks = chunkText(textContent, 1000);
                const finalRecords = [];
                
                for (let i = 0; i < chunks.length; i++) {
                    if (!chunks[i].trim()) continue;
                    const values = await getEmbedding(chunks[i]);
                    // O ID Fixo garante que ele não duplica textos já existentes
                    const safeId = `${uniqueName.replace(/[^a-zA-Z0-9]/g, '_')}_chunk_${i}`;
                    finalRecords.push({ id: safeId, values: values, metadata: { text: chunks[i], source: `Global: ${file}` } });
                }
                if (finalRecords.length > 0) {
                    await pineconeIndex.upsert(finalRecords);
                    console.log(`📚 [Sucesso] Texto ${file} indexado no Pinecone!`);
                }
            }
        } catch (err) {
            console.error(`❌ Erro ao ler ${file}: ${err.message}`);
        }
    }
    console.log("🏁 Sincronização Global Concluída!\n");
}


// ==========================================
// ROTA 1: CHAT COM INJEÇÃO GLOBAL
// ==========================================
app.post('/api/chat', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        const { query, geminiFiles } = req.body;
        console.log(`\n🧠 Pergunta recebida: "${query}"`);

        // Pesquisa Rápida Pinecone
        let queryVectorRaw = await getEmbedding(query);
        let queryVector = [];
        for(let i=0; i<queryVectorRaw.length; i++) queryVector.push(Number(queryVectorRaw[i]));
        
        let searchResults = { matches: [] };
        try { searchResults = await pineconeIndex.query({ vector: queryVector, topK: 3, includeMetadata: true }); } catch (e) {}
        let contextText = searchResults.matches ? searchResults.matches.map(m => `[Documento: ${m.metadata?.source}]\n${m.metadata?.text}`).join('\n\n') : '';

        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const systemPrompt = `Você é o RACKS IA, um assistente técnico especializado da APROC.
Responda de forma rigorosa com base nos arquivos anexados a esta conversa e na Base de Conhecimento Rápida abaixo:
${contextText || "Sem contexto rápido."}`;

        const chatParts = [{ text: systemPrompt }, { text: `PERGUNTA: ${query}` }];

        // 🌟 AQUI ACONTECE A MÁGICA: Juntamos os ficheiros da sua pasta com os do utilizador
        const allGeminiFiles = [...(geminiFiles || []), ...GLOBAL_GEMINI_FILES];

        if (allGeminiFiles.length > 0) {
            allGeminiFiles.forEach(file => {
                if (file.mimeType && file.uri) chatParts.push({ fileData: { mimeType: file.mimeType, fileUri: file.uri } });
            });
        }

        const resultStream = await model.generateContentStream(chatParts);
        for await (const chunk of resultStream.stream) {
            res.write(`data: ${JSON.stringify({ text: chunk.text() })}\n\n`);
        }
        res.write(`data: [DONE]\n\n`);
        res.end();

    } catch (error) {
        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
        res.end();
    }
});

// Arranca o servidor IMEDIATAMENTE e faz a sincronização em segundo plano
app.listen(port, () => {
    console.log(`🚀 SERVIDOR RACKS IA PRONTO NA PORTA ${port}`);
    
    // Inicia a leitura dos PDFs sem bloquear o arranque do Render
    syncGlobalDocuments().catch(err => console.error("Falha na sincronização global:", err));
});