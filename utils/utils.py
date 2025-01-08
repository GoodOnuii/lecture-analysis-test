import os
import re
import kss
import json
import boto3
from .time_converter import TimeConverter

def get_items(bucket, folder):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
    content_list = []
    for content in response.get('Contents', []):
        key = content.get('Key', None)
        if len(key.split('/')) == 4:
            content_list.append(key)
    return content_list

def download_items(bucket, file_keys, local_dir):
    s3 = boto3.client('s3')
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    for file_key in file_keys:
        folder_name = os.path.basename(os.path.dirname(file_key))
        file_name = os.path.basename(file_key)
        local_file_path = os.path.join(local_dir, f"{folder_name}_{file_name}")
        s3.download_file(bucket, file_key, local_file_path)
        
def merge_files(directory):
    merged_data = []
    
    for file_name in os.listdir(directory):
        parts = file_name.split('_')
        absolute_start_time = int(parts[-2])
        speaker = 'teacher' if 'T' in parts[0] else 'student'
    
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
            for item in content:
                start_ms = TimeConverter.convert_time_to_milliseconds(item['start'])
                end_ms = TimeConverter.convert_time_to_milliseconds(item['end'])
                
                merged_item = {
                    "absolute_start_time": absolute_start_time + start_ms,
                    "absolute_end_time": absolute_start_time + end_ms,
                    "time_range": None,
                    "speaker": speaker,
                    "text": item["text"].strip(),
                }
                merged_data.append(merged_item)
                
    # 절대 시작 시간을 기준으로 정렬
    merged_data.sort(key=lambda x: x["absolute_start_time"])

    global_start_time = merged_data[0]["absolute_start_time"]  # 가장 이른 절대 시작 시간
    for item in merged_data:
        item["time_range"] = f"{TimeConverter.format_ms_to_xm_ys(item['absolute_start_time'] - global_start_time)} ~ {TimeConverter.format_ms_to_xm_ys(item['absolute_end_time'] - global_start_time)}"
        item['absolute_start_time'] = TimeConverter.format_ms_to_ktc(item['absolute_start_time'])
        item['absolute_end_time'] = TimeConverter.format_ms_to_ktc(item['absolute_end_time'])
    
    return merged_data

def extract_speaker(data, speaker=None):
    if speaker == None:
        return [{'time_range': item['time_range'], 'absolute_start_time': item['absolute_start_time'], 'absolute_end_time': item['absolute_end_time'], 'speaker': item['speaker'], 'text': item['text']} for item in data]
    if speaker == 'teacher':
        return [{'time_range': item['time_range'], 'absolute_start_time': item['absolute_start_time'], 'absolute_end_time': item['absolute_end_time'], 'speaker': item['speaker'], 'text': item['text']} for item in data if item['speaker'] == 'teacher']
    if speaker == 'student':
        return [{'time_range': item['time_range'], 'absolute_start_time': item['absolute_start_time'], 'absolute_end_time': item['absolute_end_time'], 'speaker': item['speaker'], 'text': item['text']} for item in data if item['speaker'] == 'student']

def split_sentences(data):
    splited_data = []
    idx = 0
    if data[0]['speaker'] == 'teacher':
        text = ' '.join([item['text'] for item in data])
        for sentence in kss.split_sentences(text, backend='fast'):
            for sent in kss.split_sentences(sentence, backend='mecab'):
                    splited_data.append({'idx': idx, 'text': sent})
                    idx += 1
        
    if data[0]['speaker'] == 'student':
        for item in data:
            for sentence in kss.split_sentences(item['text'], backend='fast'):
                for sent in kss.split_sentences(sentence, backend='mecab'):
                    splited_data.append({'idx': idx, 'text': sent})
                    idx += 1
    
    return splited_data

# def mapping_time(extracted_data, splited_data):
#     for item in splited_data:
#         time_list = []
#         text_list = item['text'].split()
#         for text in text_list:
#             for data in extracted_data:
#                 if text in data['text']:
#                     time_list.append(data['time_range'])
#                     data['text'] = data['text'].replace(text, "", 1).strip()
#                     break
#                 continue
#         time_list = list(set(time_list))
#         if len(time_list) > 1:
#             temp = []
#             for time in time_list:
#                 temp.append(TimeConverter.convert_timerange_to_milliseconds(time))
#             a_min = min([range_pair[0] for range_pair in temp])
#             b_max = max([range_pair[1] for range_pair in temp])
#             time_list = [a_min, b_max]
#         else:
#             time_list = TimeConverter.convert_timerange_to_milliseconds(time_list[0])
#         item['start'] =  TimeConverter.format_ms_to_xm_ys(time_list[0])
#         item['end'] = TimeConverter.format_ms_to_xm_ys(time_list[1])
#     return splited_data

def mapping_time(extracted_data, splited_data):
    for item in splited_data:
        timerange_list = []
        abstime_list = []
        text_list = item['text'].split()
        for text in text_list:
            for data in extracted_data:
                if text in data['text']:
                    timerange_list.append(data['time_range'])
                    abstime_list.append((data['absolute_start_time'], data['absolute_end_time']))
                    data['text'] = data['text'].replace(text, "", 1).strip()
                    break
                continue
        time_list = list(set(timerange_list))
        abstime_list = list(set(abstime_list))
        if len(time_list) > 1:
            temp = []
            for time in timerange_list:
                temp.append(TimeConverter.convert_timerange_to_milliseconds(time))
            a_min = min([range_pair[0] for range_pair in temp])
            b_max = max([range_pair[1] for range_pair in temp])
            time = TimeConverter.format_ms_to_xm_ys_range([a_min, b_max])
        else:
            time = time_list[0]
        item['time'] = time
        item['start'] = abstime_list[0][0]
        item['end'] = abstime_list[0][1]
    return splited_data

def split_with_overlap(data, chunk_size, overlap):
    result = []
    step = chunk_size - overlap
    for i in range(0, len(data), step):
        result.append(data[i:i + chunk_size].to_dict(orient='records'))
        if i + chunk_size >= len(data):
            break
    return result

def extract_question_indices(data):
    all_indices = []
    for entry in data:
        json_text = re.search(r'```json\n(.*?)\n```', entry, re.DOTALL).group(1)
        parsed_json = json.loads(json_text)
        all_indices.extend(parsed_json[0]["idx"])
    return sorted(set(all_indices))

def get_question_context_v1(df, question_indices, speaker, range_size):
    
    target_indices = df[df[f'{speaker}_idx'].isin(question_indices)].index.tolist()
    data = df.to_dict(orient='records')
    grouped_result = []
    for idx in target_indices:
        question = data[idx][f'{speaker}_text']
        start_idx = max(0, idx - range_size)
        end_idx = min(len(data), idx + range_size + 1)
        contexts = data[start_idx:end_idx]
        
        context = []
        for item in contexts:
            if item['teacher_idx'] != None:
                context.append({'time': item['time'], 'teacher_text': item['teacher_text']})
            else:
                context.append({'time': item['time'], 'student_text': item['student_text']})
        
        grouped_result.append({'idx': idx, 'question': question, 'context': context})
    return grouped_result

def extract_True_indices(results):
    
   # True인 항목의 인덱스만 리스트로 반환
    indices = []
    for res in results:
        # 결과값을 소문자로 변환하여 비교 (대소문자 구분 없이)
        if res['result'].lower() == 'true':
            indices.append(res['idx'])
    return indices

def get_question_context_v2(df, learning_question_indices, speaker, range_size):
    
    #target_indices = df[df[f'{speaker}_idx'].isin(question_indices)].index.tolist()
    data = df.to_dict(orient='records')
    grouped_result = []
    for idx in learning_question_indices:
        question = data[idx][f'{speaker}_text']
        start_idx = max(0, idx - range_size)
        end_idx = min(len(data), idx + range_size + 1)
        contexts = data[start_idx:end_idx]
        
        context = []
        for item in contexts:
            if item['teacher_idx'] != None:
                context.append({'time': item['time'], 'teacher_text': item['teacher_text']})
            else:
                context.append({'time': item['time'], 'student_text': item['student_text']})
        
        grouped_result.append({'idx': idx, 'question': question, 'context': context})
    return grouped_result
