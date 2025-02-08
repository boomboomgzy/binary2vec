from utils import init_nodevector
import os
import sys





if __name__=='__main__':

    prop_threads=50
    root=r''
    save_dir=os.path.join(root,'dataset')
    save_dir=r''


    homoG_save_dir=os.path.join(save_dir,'sa_homoG')
    ir_programl_dir=os.path.join(save_dir,'sa_programl')
    vocab_dir=r''

    corpus_model_path=os.path.join(vocab_dir,'ir_corpus.model')
    corpus_vec_path=os.path.join(vocab_dir,'ir_corpus.vector')
    vector_log_file=os.path.join(vocab_dir,'vetor_log.txt')


    init_nodevector(ir_programl_dir,homoG_save_dir,corpus_model_path,corpus_vec_path,prop_threads,vector_log_file)
    #初始化后要重新构建数据集
