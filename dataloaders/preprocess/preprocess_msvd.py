import json
import pickle
import os

def preprocess_msvd_captions(data_dir):
    all_captions = {}
    
    # 汇总 train, val, test 的 json
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(data_dir, f'msvd_{split}.json')
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping.")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                v_id = item['video_id']
                # 将每个字幕字符串转换为单词列表 (tokens)
                # 使用 split() 简单切分，并转换为小写
                caps = [c.lower().split() for c in item['caption']]
                all_captions[v_id] = caps
    
    output_path = os.path.join(data_dir, 'raw-captions.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_captions, f)
    
    print(f"Successfully created {output_path}")
    print(f"Total video IDs processed: {len(all_captions)}")

if __name__ == "__main__":
    msvd_data_dir = "/data2/hxj/data/MSVD/desc_files"
    preprocess_msvd_captions(msvd_data_dir)

