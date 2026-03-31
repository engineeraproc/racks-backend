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

app.use(cors({ 
    origin: '*',
    methods: ['GET', 'POST', 'DELETE', 'PUT', 'PATCH', 'OPTIONS']
}));
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

// 2. ROTA DE CHAT (PERSONALIDADE E DIRETRIZES ATUALIZADAS)
app.post('/api/chat', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        const { query, history } = req.body;
        console.log(`\n🧠 Pergunta: "${query}"`);
        
        let contextText = '';
        if (pineconeIndex) {
            try {
                let queryVectorRaw = await getEmbedding(query);
                if (queryVectorRaw && queryVectorRaw.length > 0) {
                    let queryVector = queryVectorRaw.map(Number);
                    let searchResults = await pineconeIndex.query({ vector: queryVector, topK: 3, includeMetadata: true });
                    contextText = searchResults.matches ? searchResults.matches.map(m => `[Contexto extraído]\n${m.metadata?.text}`).join('\n\n') : '';
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
        const chatParts = [];

        allGeminiFiles.forEach(file => { 
            if (file.mimeType && file.uri) chatParts.push({ fileData: { mimeType: file.mimeType, fileUri: file.uri } }); 
        });

        if (history && history.length > 0) {
            const historyText = history.map(h => `${h.role === 'user' ? 'Usuário' : 'RACKS IA'}: ${h.content}`).join('\n');
            chatParts.push({ text: `HISTÓRICO RECENTE DA CONVERSA:\n${historyText}\n\n---\n\n` });
        }

        chatParts.push({ text: `DIRETRIZES ABSOLUTAS DE COMPORTAMENTO DA RACKS IA:
1. IDENTIDADE E SIGILO MÁXIMO: Você é a RACKS IA, a inteligência artificial oficial de engenharia da APROC. Aja como se possuísse todo o conhecimento técnico nativamente na sua mente. É ESTRITAMENTE PROIBIDO mencionar palavras como "arquivos", "documentos", "PDFs", "anexos", "upload" ou "base de dados". NUNCA cite o nome de nenhum arquivo nas suas respostas.
2. INTERAÇÃO SOCIAL HUMANA: Você DEVE responder de forma natural, amigável e prestativa a saudações ou interações casuais (ex: Oi, Olá, Bom dia, Tudo bem?, Quem é você?). Se o usuário disser "Oi", responda com simpatia sem dizer que falta contexto técnico.
3. RIGOR TÉCNICO E BUSCA CEGA: Para responder a perguntas técnicas, realize uma busca exaustiva e silenciosa em todo o conhecimento que lhe foi injetado (os arquivos que você recebeu, mas que não deve mencionar). A informação solicitada pode estar no primeiro parágrafo ou oculta no meio de qualquer parte do seu conhecimento. Procure o termo exato.
4. REGRA DE DESCONHECIMENTO TÉCNICO (MUITO IMPORTANTE): Se a pergunta for técnica e, após procurar profundamente, você não encontrar a resposta exata no seu conhecimento, você NÃO deve pedir desculpas nem dar explicações sobre faltar documentos. Você DEVE responder APENAS E EXATAMENTE com esta frase: "A RACKS IA ainda não tem informações concretas sobre [Insira o Assunto Aqui]."

Conhecimento adicional injetado: ${contextText}` });

        chatParts.push({ text: `\n\nAGORA RESPONDA À SEGUINTE INTERAÇÃO/PERGUNTA DO USUÁRIO:\nUSUÁRIO: ${query}` });

        const resultStream = await model.generateContentStream(chatParts);
        for await (const chunk of resultStream.stream) {
            try { res.write(`data: ${JSON.stringify({ text: chunk.text() })}\n\n`); } catch(e) {}
        }
        res.write(`data: [DONE]\n\n`);
        res.end();
    } catch (error) {
        let msg = error.message;
        if (msg.includes('429') || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('overloaded')) {
            msg = "⚠️ O servidor atingiu o limite de consultas rápidas. Por favor, aguarde 2 minutos e tente novamente.";
        }
        res.write(`data: ${JSON.stringify({ error: msg })}\n\n`);
        res.end();
    }
});

// 3. O RADAR DA NUVEM
app.get('/api/status', async (req, res) => {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.setHeader('Surrogate-Control', 'no-store');

    try {
        let processing = 0;
        let active = 0;
        let total = 0;
        let filesList = [];
        let pageToken;

        try {
            do {
                const listResult = await fileManager.listFiles({ pageToken });
                if (listResult.files) {
                    for (const f of listResult.files) {
                        total++;
                        if (f.state === 'ACTIVE') active++;
                        if (f.state === 'PROCESSING') processing++;
                        
                        filesList.push({
                            name: f.name,
                            displayName: f.displayName,
                            uri: f.uri,
                            mimeType: f.mimeType,
                            state: f.state,
                            updateTime: f.updateTime
                        });
                    }
                }
                pageToken = listResult.nextPageToken;
            } while (pageToken);
            
            filesList.sort((a, b) => new Date(b.updateTime) - new Date(a.updateTime));
        } catch(e) {}

        res.json({ active, processing, total, files: filesList });
    } catch(err) { res.status(500).json({ error: err.message }); }
});

// 4. AUDITORIA DE LEITURA (COM AVISO CORRIGIDO)
app.post('/api/analyze-file', async (req, res) => {
    try {
        const { uri, mimeType } = req.body;
        if (!uri) throw new Error("URI não fornecido.");

        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const prompt = `Atue como um analista de extração de dados. O documento em anexo precisa ser lido e compreendido.
Por favor, retorne APENAS um objeto JSON estrito com o seguinte formato, sem formatação markdown:
{
  "summary": "Um resumo claro de 2 a 3 frases sobre o conteúdo principal do documento",
  "score": <um número inteiro de 0 a 100 avaliando o quão fácil, legível e bem estruturado é o texto para você ler e extrair informações>,
  "readability": "Uma frase curta diagnosticando a qualidade do ficheiro (ex: 'O texto é perfeitamente legível e estruturado nativamente')"
}`;

        const result = await model.generateContent([
            { fileData: { mimeType: mimeType, fileUri: uri } },
            { text: prompt }
        ]);

        let textResponse = result.response.text();
        textResponse = textResponse.replace(/```json/g, '').replace(/```/g, '').trim();
        
        let analysis;
        try {
            analysis = JSON.parse(textResponse);
        } catch (e) {
            analysis = { summary: textResponse.substring(0, 150) + "...", score: 50, readability: "O arquivo foi lido, mas a formatação gerou conflito na extração." };
        }

        res.json(analysis);
    } catch (error) {
        let msg = error.message;
        if (msg.includes('429') || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('overloaded')) {
            msg = "O servidor atingiu o limite de auditorias. Aguarde 2 minutos antes de auditar outro documento.";
        }
        res.status(500).json({ error: msg });
    }
});

// 5. APAGAR 1 FICHEIRO
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

// 6. BOTÃO NUCLEAR
app.delete('/api/clear-cloud', async (req, res) => {
    try {
        let count = 0;
        let pageToken;
        do {
            try {
                const listResult = await fileManager.listFiles({ pageToken });
                if (listResult.files) {
                    for (const file of listResult.files) {
                        try {
                            await fileManager.deleteFile(file.name);
                            count++;
                        } catch(delErr) {}
                    }
                }
                pageToken = listResult.nextPageToken;
            } catch(e) { break; }
        } while (pageToken);

        try { if(pineconeIndex) await pineconeIndex.deleteAll(); } catch(e) {}
        
        res.json({ success: true, count });
    } catch (error) { res.status(500).json({ error: error.message }); }
});

app.listen(port, () => console.log(`🚀 SERVIDOR COM COMPORTAMENTO HUMANO PRONTO NA PORTA ${port}`));