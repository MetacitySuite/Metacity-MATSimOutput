import pandas as pd
import numpy as np
import os
import shutil
import json
from multiprocessing import Pool

INPUT_PATH = "./../output/agents/"
OUTPUT_PATH = "./../output/events/"
CHUNK_SIZE = 500

def clear_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)



class Chunker:
    def __init__(self, agents_path = INPUT_PATH, events_path = OUTPUT_PATH):
        self.input_path = agents_path
        self.folders = os.listdir(agents_path)
        self.output_path = events_path


    def __call__(self, chunk_size = CHUNK_SIZE):
        clear_directory(self.output_path)
        print(self.folders)
        for agent_type in self.folders:
            files = np.array(os.listdir(self.input_path+agent_type))
            transport_map = {}
            print("# of files to chunk:", len(files))
            if(len(files) > 0):
                chunks_num = int(len(files) / chunk_size)
                if(len(files) % chunk_size > 0):
                    chunks_num += 1
    
                print("# of chunks:", chunks_num)
                file_chunks = np.array_split(files, chunks_num)
    
                args = list()
                for i,f in enumerate(file_chunks):
                    args.append([self.input_path+agent_type+"/",f,i, agent_type])
    
    
                with Pool(int(min(chunks_num, os.cpu_count()))) as pool:
                    results = pool.map(self.concat_files, args)
    
                pool.close()
                pool.join()
    
                if(agent_type != "agent"):
                    # concat results
                    for r in results:
                        transport_map.update(r)
                    # save transport map
                    with open(self.output_path+agent_type+'_map.json', 'w') as f:
                        json.dump(transport_map,f)


    def concat_files(self, args):
        path, files, chunk_i, agent_type = args
        df = pd.DataFrame()
        chunk_map = {}
        if(agent_type != "agent"):
            ids = [f.split('/')[-1].split('.')[0] for f in files] #stracting ids
            for i in ids:
                chunk_map[i] = chunk_i

        for file in files:
            # load file_chunks
            agent = pd.read_json(path+file, lines=True, orient='records')
            df = df.append(agent)

        # save to file
        del agent
        if not os.path.exists(self.output_path+agent_type):
            os.makedirs(self.output_path+agent_type)

        df.to_json(self.output_path+agent_type+"/"+str(chunk_i)+".json", lines=True, orient='records') 
        del df
        return chunk_map