word_vec_path="../../glove.txt"
word_vec_file=open(word_vec_path,'r',encoding='utf-8')
all_vecs=word_vec_file.readlines()
# for i in range(3):
#     tmp_line=word_vec_file.readline()
#     print(tmp_line)
#     print
print(all_vecs[100])
word_vec_file.close()