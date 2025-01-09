from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import boto3
import os
import json
import shutil
import pandas as pd
import importlib
from utils import utils
from prompt import question_checker, question_classifier, teacher_digging, student_concretizing
import time 
import asyncio

def setup_llm():
    load_dotenv()
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    return ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=DEEPSEEK_API_KEY, 
        openai_api_base='https://api.deepseek.com',
    )

def process_raw_data(subject, room_id):
    # Get and process files
    file_keys = utils.get_items('pagecall-text', f'{subject}/{room_id}')
    utils.download_items('pagecall-text', file_keys, './downloads')
    
    raw_data = utils.merge_files('./downloads')
    shutil.rmtree('./downloads')
    
    return raw_data

def create_dataframe(raw_data):
    # Process teacher data
    teacher_extracted_data = utils.extract_speaker(raw_data, speaker='teacher')
    teacher_splited_data = utils.split_sentences(teacher_extracted_data)
    teacher_splited_data = utils.mapping_time(teacher_extracted_data, teacher_splited_data)
    teacher_df = pd.DataFrame(teacher_splited_data).rename(columns={"idx": "teacher_idx", "text": "teacher_text"})
    
    # Process student data
    student_extracted_data = utils.extract_speaker(raw_data, speaker='student')
    student_splited_data = utils.split_sentences(student_extracted_data)
    student_splited_data = utils.mapping_time(student_extracted_data, student_splited_data)
    student_df = pd.DataFrame(student_splited_data).rename(columns={"idx": "student_idx", "text": "student_text"})
    
    # Combine and process final dataframe
    df = pd.concat([teacher_df, student_df], ignore_index=True)
    df = df.sort_values(by=["start", "teacher_idx", "student_idx"]).reset_index(drop=True)
    df = df.astype({'teacher_idx': 'Int64', 'student_idx': 'Int64'})
    df = df[['start', 'end', 'teacher_idx', 'student_idx', 'time', 'teacher_text', 'student_text']]
    
    return df

def save_results(teacher_context, student_context, room_id):
    # Save teacher results
    with open(f"{room_id}_선생님.txt", "w", encoding="utf-8-sig") as file:
        json.dump(teacher_context, file, ensure_ascii=False, indent=4)
    
    # Save student results
    with open(f"{room_id}_학생.txt", "w", encoding="utf-8-sig") as file:
        json.dump(student_context, file, ensure_ascii=False, indent=4)
 


async def question_check_async(chunks_with_overlap, subject, user, LLM):
    system_prompt = question_checker.QuestionChecker(subject=subject, user=user).prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('user', "{user_message}")
    ])
    chain = prompt | LLM | StrOutputParser()
    
    results = await chain.abatch([{"user_message": chunk} for chunk in chunks_with_overlap])
    return utils.extract_question_indices(results)

async def learning_question_check_async(question_context, subject, user, LLM):
    system_prompt = question_classifier.QuestionClassifier(subject=subject, user=user).prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('user', "{user_message}")
    ])
    chain = prompt | LLM | JsonOutputParser()
    
    results = await chain.abatch([{"user_message": chunk} for chunk in question_context])
    return utils.extract_True_indices(results)

async def analyze_teacher_questions_async(df, chunks_with_overlap, subject, LLM):

    # Initial question check
    t_start = time.time()
    question_indices = await question_check_async(chunks_with_overlap, subject, '선생님', LLM)
    question_context = utils.get_question_context_v1(df, question_indices, 'teacher', 5)
    t_end = time.time()
    print(f"Teacher_question analyses in {t_end - t_start:.2f} seconds")
    # Learning question check
    t_start = time.time()
    learning_indices = await learning_question_check_async(question_context, subject, '선생님', LLM)
    learning_context = utils.get_question_context_v2(df, learning_indices, 'teacher', 5)
    t_end = time.time()
    print(f"Teacher_learning_question analyses in {t_end - t_start:.2f} seconds")
    
    # Digging question check
    system_prompt = teacher_digging.Digging(subject).prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('user', "{user_message}")
    ])
    chain = prompt | LLM | JsonOutputParser()
    
    results = await chain.abatch([{"user_message": chunk} for chunk in learning_context])
    digging_indices = utils.extract_True_indices(results)
    
    return utils.get_question_context_v2(df, digging_indices, 'teacher', 5)

async def analyze_student_questions_async(df, chunks_with_overlap, subject, LLM):
    # Initial question check
    t_start = time.time()
    question_indices = await question_check_async(chunks_with_overlap, subject, '학생', LLM)
    question_context = utils.get_question_context_v1(df, question_indices, 'student', 5)
    t_end = time.time()
    print(f"Student_question analyses in {t_end - t_start:.2f} seconds")
    # Learning question check
    t_start = time.time()
    learning_indices = await learning_question_check_async(question_context, subject, '학생', LLM)
    learning_context = utils.get_question_context_v2(df, learning_indices, 'student', 5)
    t_end = time.time()
    print(f"Student_question analyses in {t_end - t_start:.2f} seconds")
    # Concretizing question check
    system_prompt = student_concretizing.concretizing(subject).prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('user', "{user_message}")
    ])
    chain = prompt | LLM | JsonOutputParser()
    
    results = await chain.abatch([{"user_message": chunk} for chunk in learning_context])
    concretizing_indices = utils.extract_True_indices(results)
    
    return utils.get_question_context_v2(df, concretizing_indices, 'student', 5)

async def main_async(subject, room_id):
    # Setup
    LLM = setup_llm()
    
    # Process data
    raw_data = process_raw_data(subject, room_id)
    df = create_dataframe(raw_data)
    
    # Prepare teacher data
    teacher_df = df[df['teacher_text'].notnull()].drop(columns=['student_text', 'student_idx', 'start', 'end', 'time'])\
        .rename(columns={"teacher_idx": "idx", "teacher_text": "text"}).reset_index(drop=True)
    teacher_chunks = utils.split_with_overlap(teacher_df, chunk_size=30, overlap=5)
    
    # Prepare student data
    student_df = df[df['student_text'].notnull()].drop(columns=['teacher_text', 'teacher_idx', 'start', 'end', 'time'])\
        .rename(columns={"student_idx": "idx", "student_text": "text"}).reset_index(drop=True)
    student_chunks = utils.split_with_overlap(student_df, chunk_size=30, overlap=5)
    
    # Start timing
    t_start = time.time()
    
    # Run analyses concurrently
    teacher_context, student_context = await asyncio.gather(
        analyze_teacher_questions_async(df, teacher_chunks, subject, LLM),
        analyze_student_questions_async(df, student_chunks, subject, LLM)
    )
    
    t_end = time.time()
    print(f"Finished all analyses in {t_end - t_start:.2f} seconds")
    
    # Save results
    save_results(teacher_context, student_context, room_id)
    
if __name__ == "__main__":
    subject = "수학"  # Example subject
    room_id = "674e58163925f28f6caf4fa0"  # Example room_id
    asyncio.run(main_async(subject, room_id))
    
    
    