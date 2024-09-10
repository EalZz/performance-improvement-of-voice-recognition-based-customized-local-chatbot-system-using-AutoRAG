# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langserve import RemoteRunnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import Document
import speech_recognition as sr
import pyttsx3

# 원격 LLM 초기화
remote_llm = RemoteRunnable("https://holy-integral-redfish.ngrok-free.app/llm/")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# KoReRanker 모델 초기화
model_path = "Dongjin-kr/ko-reranker"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 음성 인식 초기화
recognizer = sr.Recognizer()

# 음성 합성 초기화
engine = pyttsx3.init()

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore, texts

def bm25_retriever(query, docs, k=5):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(doc_scores)[::-1][:k]
    return [docs[i] for i in top_n]

def rerank_with_koreranker(query, docs):
    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())
    reranked_docs = [doc for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked_docs[:min(3, len(reranked_docs))]

answer_prompt = PromptTemplate.from_template(
    "주어진 단락을 이용하여 다음 질문에 100단어 이내로 간결하게 답하시오. 반드시 100단어를 초과하지 마세요. 답변이 끝나면 반드시 '<END_OF_RESPONSE>'를 추가하세요:\n질문: {question}\n\n단락: {context}\n\n답변:"
)

def create_rag_chain(vectorstore, docs):
    def retrieve_and_rerank(query):
        initial_docs = bm25_retriever(query, docs)
        reranked_docs = rerank_with_koreranker(query, initial_docs)
        return reranked_docs

    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            context=RunnablePassthrough() | retrieve_and_rerank,
        )
        | {
            "context": lambda x: "\n".join([doc.page_content for doc in x["context"]]),
            "question": lambda x: x["question"],
        }
        | answer_prompt
        | remote_llm
    )
    return rag_chain

def truncate_response(response, max_words=100):
    response = response.split('<END_OF_RESPONSE>')[0].strip()
    words = response.split()
    if len(words) <= max_words:
        return response
    truncated = ' '.join(words[:max_words])
    last_sentence_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_sentence_end != -1:
        truncated = truncated[:last_sentence_end+1]
    return truncated.strip()

def get_audio_input():
    with sr.Microphone() as source:
        print("음성으로 질문해주세요...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        print("인식된 질문:", text)
        return text
    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
        return None
    except sr.RequestError as e:
        print("음성 인식 서비스에 오류가 발생했습니다; {0}".format(e))
        return None

def speak_response(text):
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    pdf_path = "./컴소학과v5_3.pdf"
    vectorstore, docs = create_vector_store(pdf_path)
    rag_chain = create_rag_chain(vectorstore, docs)

    while True:
        print("1. 텍스트로 질문하기")
        print("2. 음성으로 질문하기")
        print("3. 종료")
        choice = input("선택해주세요 (1/2/3): ")

        if choice == '3':
            break

        if choice == '1':
            user_input = input("질문을 입력하세요: ")
        elif choice == '2':
            user_input = get_audio_input()
            if user_input is None:
                continue
        else:
            print("잘못된 선택입니다. 다시 선택해주세요.")
            continue

        try:
            rag_response = rag_chain.invoke(user_input)
            truncated_response = truncate_response(rag_response.content)
            print("응답:", truncated_response)
            speak_response(truncated_response)
        except Exception as e:
            print("오류 발생:", str(e))
