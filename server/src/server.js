import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import path from "path";
import fs from "fs/promises";
import chat from "./chat.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json()); // 启用JSON body解析

// 会话状态存储
const sessions = new Map();

// 增强版multer配置
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(process.cwd(), "server/uploads");
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const sessionId = uuidv4();
    const ext = path.extname(file.originalname);
    cb(null, `${sessionId}${ext}`); // 使用UUID作为文件名
  },
});

const fileFilter = (req, file, cb) => {
  if (file.mimetype === "application/pdf") {
    cb(null, true);
  } else {
    cb(new Error("仅支持PDF文件"), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB限制
});

// 文件上传接口
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "未接收到文件" });
    }

    const sessionId = path.parse(req.file.filename).name;
    const filePath = req.file.path;

    // 初始化会话状态
    sessions.set(sessionId, {
      status: "processing",
      filePath,
      vectorStore: null,
      error: null,
    });

    // 异步处理文档
    setTimeout(async () => {
      try {
        // 调用chat函数初始化处理
        await chat(filePath); // 触发向量存储创建

        // 加载向量存储用于后续查询
        const embeddings = new OpenAIEmbeddings({
          openAIApiKey: process.env.OPENAI_API_KEY,
        });
        const vectorStore = await FaissStore.load(
          path.join(process.cwd(), "server/vector_store"),
          embeddings
        );

        sessions.set(sessionId, {
          status: "ready",
          filePath,
          vectorStore,
        });
      } catch (error) {
        sessions.set(sessionId, {
          status: "error",
          error: error.message,
        });
      }
    }, 0);

    res.json({
      sessionId,
      statusUrl: `/status/${sessionId}`,
      chatUrl: `/chat`,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// 状态查询接口
app.get("/status/:sessionId", (req, res) => {
  const session = sessions.get(req.params.sessionId);
  if (!session) {
    return res.status(404).json({ error: "会话不存在" });
  }
  res.json({
    status: session.status,
    ...(session.error && { error: session.error }),
  });
});

// 问答接口（改为POST）
app.post("/chat", async (req, res) => {
  try {
    const { sessionId, question } = req.body;

    if (!sessionId || !question) {
      return res.status(400).json({ error: "缺少必要参数" });
    }

    const session = sessions.get(sessionId);
    if (!session) {
      return res.status(404).json({ error: "会话已过期或不存在" });
    }

    if (session.status !== "ready") {
      return res.status(425).json({
        error: "文档处理中，请稍后再试",
        statusUrl: `/status/${sessionId}`,
      });
    }

    // 调用chat函数进行问答
    const response = await chat(null, question); // 使用已加载的vectorStore

    res.json({
      answer: response.text,
      sessionId,
    });
  } catch (error) {
    res.status(500).json({
      error: "问答服务暂时不可用",
      detail: error.message,
    });
  }
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  // 初始化目录
  fs.mkdir(path.join(process.cwd(), "server/vector_store"), {
    recursive: true,
  });
});
