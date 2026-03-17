# -*- coding: utf-8 -*-
"""
RAG Chat: load vector store + Llama2, answer questions based on your PDFs.
Run: python chat.py
Uses only langchain_community retriever + LLM (no langchain.chains).
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline, Ollama
import torch

import config

RAG_PROMPT = """请仅根据下面的「参考内容」回答问题。
规则：只使用参考内容中的信息回答；参考中没有的内容不要编造、不要推测。若参考内容与问题无关或不足以回答问题，你必须明确回复「根据上述参考无法回答该问题」或「参考中没有相关信息」，不要编造答案。

参考内容：
{context}

问题：{question}

回答："""

def get_llm(model_name: str = None, use_finetuned: bool = False):
    """Load LLM: Ollama or Hugging Face Llama2. If model_name is given, use HF/transformers with that model (ignore USE_OLLAMA)."""
    if config.USE_OLLAMA and model_name is None:
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,   # lower to reduce hallucination and favor "cannot answer" when uncertain
            num_predict=512,
        )
    # Hugging Face / local transformers model
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_name = model_name or config.LLAMA2_MODEL
    if use_finetuned and config.FINETUNE_OUTPUT_DIR.exists():
        model_path = str(config.FINETUNE_OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=config.HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HF_TOKEN,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HF_TOKEN,
        )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,   # lower to reduce hallucination
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_rag_chain(use_finetuned: bool = False, model_name: str = None):
    """Build RAG: retriever + LLM, implemented without langchain.chains."""
    config.ensure_dirs()
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(model_name=model_name, use_finetuned=use_finetuned)

    class RAGWrapper:
        def invoke(self, inp):
            question = inp.get("query", inp.get("question", ""))
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            prompt_text = RAG_PROMPT.format(context=context, question=question)
            result = llm.invoke(prompt_text)
            if not isinstance(result, str):
                result = getattr(result, "content", str(result))
            return {"result": result, "source_documents": docs}

    return RAGWrapper()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG 对话（基于 PDF 向量库 + Llama2）")
    parser.add_argument("--finetuned", action="store_true", help="使用 finetuned_llama/ 中的微调模型（需 USE_OLLAMA=false）")
    parser.add_argument("--model", type=str, default=None, help="指定 HF 模型 id 或本地路径，如 NousResearch/Llama-2-7b-chat-hf（需 USE_OLLAMA=false）")
    args = parser.parse_args()
    qa = get_rag_chain(use_finetuned=args.finetuned, model_name=args.model)
    print("基于 PDF 的 RAG 对话（输入 'q' 退出）\n")
    while True:
        try:
            question = input("你: ").strip()
            if not question or question.lower() == "q":
                break
            out = qa.invoke({"query": question})
            print("助手:", out["result"])
            if out.get("source_documents"):
                print("  [参考片段数:", len(out["source_documents"]), "]")
            print()
        except KeyboardInterrupt:
            break
    print("再见。")

if __name__ == "__main__":
    main()
