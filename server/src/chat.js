import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import path from "path";
import NodeCache from "node-cache";

// 初始化缓存（内存缓存）
const cache = new NodeCache({ stdTTL: 600, checkperiod: 120 });

// 异步处理队列
const processingQueue = new Map(); // 简易内存队列替代Bull

// FAISS配置
const VECTOR_STORE_DIR = path.join(process.cwd(), "server/vector_store");

const initializeVectorStore = async (splitDocs, embeddings) => {
  try {
    return await FaissStore.load(VECTOR_STORE_DIR, embeddings);
  } catch (error) {
    const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
    await vectorStore.save(VECTOR_STORE_DIR);
    return vectorStore;
  }
};

// 带缓存的问答处理
const processQueryWithCache = async (query, vectorStore) => {
  const cachedResponse = cache.get(query);
  if (cachedResponse) return cachedResponse;

  const model = new ChatOpenAI({
    modelName: "gpt-4.0",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const template = `Use the context to answer concisely:
{context}
Question: {question}
Answer:`;

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    prompt: PromptTemplate.fromTemplate(template),
  });

  const response = await chain.call({ query });
  cache.set(query, response);
  return response;
};

// 主处理函数
export const chat = async (filePath, query) => {
  // 异步处理标记
  if (!query) {
    if (!processingQueue.has(filePath)) {
      processingQueue.set(
        filePath,
        (async () => {
          try {
            const loader = new PDFLoader(filePath);
            const data = await loader.load();

            const textSplitter = new RecursiveCharacterTextSplitter({
              chunkSize: 500,
              chunkOverlap: 50,
            });

            const splitDocs = await textSplitter.splitDocuments(data);
            const embeddings = new OpenAIEmbeddings({
              openAIApiKey: process.env.OPENAI_API_KEY,
            });

            await initializeVectorStore(splitDocs, embeddings);
          } finally {
            processingQueue.delete(filePath);
          }
        })()
      );
    }
    return { status: "processing" };
  }

  // 实际查询处理
  try {
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const vectorStore = await FaissStore.load(VECTOR_STORE_DIR, embeddings);

    return await processQueryWithCache(query, vectorStore);
  } catch (error) {
    console.error("Processing error:", error);
    return { error: "系统正在初始化，请稍后再试" };
  }
};
