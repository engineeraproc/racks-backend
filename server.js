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
let pc, pineconeIndex;

try {
    pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    pineconeIndex = pc.index(process.env.PINECONE_INDEX);
} catch (e) { console.log("Aviso: Pinecone não configurado."); }

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

async function getEmbedding(text) {
    try {
        if (!text) return [];
        const url = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${process.env.GEMINI_API_KEY}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: "models/text-embedding-004", content: { parts: [{ text: text }] } })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error.message);
        return data.embedding?.values || [];
    } catch (err) { return []; }
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
        } else { currentChunk += sentence + '. '; }
    }
    if (currentChunk.trim()) chunks.push(currentChunk.trim());
    return chunks;
}

// 1. ROTA DE UPLOAD
app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) throw new Error("Nenhum ficheiro recebido.");
        const file = req.file;
        const ext = path.extname(file.originalname).toLowerCase();
        
        const tempFilePath = path.join(os.tmpdir(), `${Date.now()}_${file.originalname}`);
        fs.writeFileSync(tempFilePath, file.buffer);

        if (ext === '.pdf' || ['.xlsx', '.xls', '.csv'].includes(ext)) {
            let fileToUpload = tempFilePath;
            let mimeType = 'application/pdf';

            if (['.xlsx', '.xls', '.csv'].includes(ext)) {
                const workbook = xlsx.readFile(tempFilePath);
                const csvData = xlsx.utils.sheet_to_csv(workbook.Sheets[workbook.SheetNames[0]]);
                fileToUpload = path.join(os.tmpdir(), `temp_csv_${Date.now()}.csv`);
                fs.writeFileSync(fileToUpload, csvData);
                mimeType = 'text/csv';
            }

            const uploadResult = await fileManager.uploadFile(fileToUpload, { mimeType: mimeType, displayName: file.originalname });
            res.json({ method: 'gemini_file', uri: uploadResult.file.uri, mimeType: mimeType });
        } 
        else if (ext === '.txt') {
            if(!pineconeIndex) throw new Error("Pinecone inativo.");
            const textContent = file.buffer.toString('utf-8');
            const chunks = chunkText(textContent, 1000);
            const finalRecords = [];
            for (let i = 0; i < chunks.length; i++) {
                if (!chunks[i].trim()) continue;
                const values = await getEmbedding(chunks[i]);
                if(values.length > 0) finalRecords.push({ id: `${Date.now()}_${i}`, values: values, metadata: { text: chunks[i], source: file.originalname } });
            }
            if (finalRecords.length > 0) await pineconeIndex.upsert(finalRecords);
            res.json({ method: 'pinecone' });
        } else { throw new Error("Formato não suportado"); }
    } catch (error) { res.status(500).json({ error: error.message }); }
});

// 2. ROTA DE CHAT
app.post('/api/chat', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        const { query } = req.body;
        let contextText = '';
        
        if (pineconeIndex) {
            try {
                let queryVectorRaw = await getEmbedding(query);
                if (queryVectorRaw && queryVectorRaw.length > 0) {
                    let queryVector = queryVectorRaw.map(Number);
                    let searchResults = await pineconeIndex.query({ vector: queryVector, topK: 3, includeMetadata: true });
                    contextText = searchResults.matches ? searchResults.matches.map(m => `[Doc: ${m.metadata?.source}]\n${m.metadata?.text}`).join('\n\n') : '';
                }
            } catch (e) { console.log("Pinecone ignorado."); }
        }

        let allGeminiFiles = [];
        try {
            const listResult = await fileManager.listFiles();
            if (listResult.files) {
                const activeFiles = listResult.files.filter(f => f.state === 'ACTIVE');
                activeFiles.sort((a, b) => new Date(b.updateTime) - new Date(a.updateTime));
                allGeminiFiles = activeFiles.slice(0, 30).map(f => ({ uri: f.uri, mimeType: f.mimeType }));
            }
        } catch(e) { }

        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const systemPrompt = `Você é o RACKS IA. Responda com base nos arquivos anexados e no contexto rápido: ${contextText || "Sem contexto extra."}`;

        const chatParts = [{ text: systemPrompt }, { text: `PERGUNTA: ${query}` }];
        allGeminiFiles.forEach(file => { if (file.mimeType && file.uri) chatParts.push({ fileData: { mimeType: file.mimeType, fileUri: file.uri } }); });

        const resultStream = await model.generateContentStream(chatParts);
        for await (const chunk of resultStream.stream) {
            try { res.write(`data: ${JSON.stringify({ text: chunk.text() })}\n\n`); } catch(e) {}
        }
        res.write(`data: [DONE]\n\n`);
        res.end();
    } catch (error) {
        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
        res.end();
    }
});

// 3. O RADAR DA NUVEM (NOVO)
app.get('/api/status', async (req, res) => {
    try {
        let processing = 0;
        let active = 0;
        let total = 0;
        try {
            const listResult = await fileManager.listFiles();
            if (listResult.files) {
                total = listResult.files.length;
                active = listResult.files.filter(f => f.state === 'ACTIVE').length;
                processing = listResult.files.filter(f => f.state === 'PROCESSING').length;
            }
        } catch(e) {}
        res.json({ active, processing, total });
    } catch(err) { res.status(500).json({ error: err.message }); }
});

// 4. APAGAR 1 FICHEIRO
app.post('/api/delete-file', async (req, res) => {
    try {
        const { uri } = req.body;
        const listResult = await fileManager.listFiles();
        if (listResult.files) {
            const fileToDelete = listResult.files.find(f => f.uri === uri);
            if (fileToDelete) {
                await fileManager.deleteFile(fileToDelete.name);
                return res.json({ success: true });
            }
        }
        res.status(404).json({ error: "Não encontrado." });
    } catch (error) { res.status(500).json({ error: error.message }); }
});

// 5. BOTÃO NUCLEAR (LIMPEZA TOTAL)
app.delete('/api/clear-cloud', async (req, res) => {
    try {
        let count = 0;
        try {
            const listResult = await fileManager.listFiles();
            if (listResult.files) {
                for (const file of listResult.files) {
                    await fileManager.deleteFile(file.name);
                    count++;
                }
            }
        } catch(e) {}
        try { if(pineconeIndex) await pineconeIndex.deleteAll(); } catch(e) {}
        res.json({ success: true, count });
    } catch (error) { res.status(500).json({ error: error.message }); }
});

app.listen(port, () => console.log(`🚀 SERVIDOR COM RADAR PRONTO NA PORTA ${port}`));