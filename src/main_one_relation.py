import sys
import tensorflow as tf
import numpy as np
from IPython import embed
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.utils.fixes import logsumexp

INF=2e9

class Naive_Bayes_Classifier:
    distrib = None
    dim_num=None
    class_num=None

    var_save_path = None
    model_mean= None
    model_var=None
    sub_rel_pos=None
    model_mean_const=None
    model_var_const=None

    my_sess=None

    def __init__(self,_sub_rel_pos,_var_save_prefix,_dim_num,_class_num):
        self.sub_rel_pos=_sub_rel_pos
        self.var_save_path=_var_save_prefix+str(_sub_rel_pos)+".ckpt"
        self.dim_num=_dim_num
        self.class_num=_class_num
        

    def fit(self, X, y):
        print("fit begins")
        sess=tf.Session()
        

        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_y = np.unique(y)
        word_features_by_class = np.array([
            [x for x, t in zip(X, y) if t == c]
            for c in unique_y])
        print("have divided word features by class")
        class_num=len(word_features_by_class)
        dim_num=len(X[0])

        mean_by_rel_id=[]
        var_by_rel_id=[]

        for sub_rel_id in range(class_num):
            words_features=word_features_by_class[sub_rel_id]
            mean, var = tf.nn.moments(tf.constant(words_features), axes=[0])
            
            mean=sess.run(mean)
            var=sess.run(var)
            mean_by_rel_id.append(mean)
            var_by_rel_id.append(var)

        mean_by_rel_id=np.array(mean_by_rel_id)
        var_by_rel_id=np.array(var_by_rel_id)
        # print((mean_by_rel_id).shape)
        # print((var_by_rel_id).shape)

        

        const_model_mean=tf.constant(mean_by_rel_id,shape=[class_num,dim_num],verify_shape=True)
        const_model_var=tf.constant(var_by_rel_id,shape=[class_num,dim_num],verify_shape=True)
        self.model_mean=tf.Variable(const_model_mean)
        self.model_var=tf.Variable(const_model_var)

        sess.run(tf.global_variables_initializer())
        sess.run(self.model_mean)
        sess.run(self.model_var)
                
        # a model
        self.distrib = tf.distributions.Normal(loc=self.model_mean, scale=tf.sqrt(self.model_var))
        
        print("distrib shape:")
        print(self.distrib.scale.shape)
        
        saver=tf.train.Saver({"model_mean":self.model_mean,"model_var":self.model_var})
        real_save_path=saver.save(sess,self.var_save_path)
        print("distribution saved in "+real_save_path)
        sess.close()

        

    def predict(self, X, _prior_num_list):
        assert self.distrib is not None
        # print(self.dist.scale.shape)
        num_classes, num_features = map(int, self.distrib.scale.shape)
        # print(nb_classes)
        # print(nb_features)

        # Conditional probabilities log P(x|c) with shape
        # (nb_samples, nb_classes)
        data_tiled=tf.tile(X, [1, num_classes])
        class_and_feature_mat=tf.reshape(data_tiled, [-1, num_classes, num_features])
        log_prob_mat=self.dist.log_prob(class_and_feature_mat)
        #each line represents a data item, sum of log p(x|c)
        condition_log_prob_mat = tf.reduce_sum(log_prob_mat,axis=2)
        # sess0=tf.Session()
        # print("after tile:")
        # print(sess0.run(tf_tile))
        # print
        # print("after reshape:")
        # print(sess0.run(new_shape_list))
        # print("log prob mat:")
        # print(sess0.run(log_prob_mat))
        # print("cond_probs(after reduce):")
        # print(sess0.run(cond_probs))

        # uniform priors
        smooth_priority_num_list=[]
        for prior_num in _prior_num_list:
            smooth_priority_num_list.append(prior_num+0.1)
        tot_prior_num=sum(smooth_priority_num_list)
        prior_prob=[]
        for prior_num in smooth_priority_num_list:
            prior_prob.append(prior_num/tot_prior_num)
        prior_lob_prob = np.log(np.array(prior_prob))
        # print("priors:")
        # print(priors)

        # posterior log probability, log P(c) + log P(x|c)
        joint_likelihood = tf.add(prior_lob_prob, condition_log_prob_mat)
        # print("joint_likelihood:")
        # print(sess0.run(joint_likelihood))

        # normalize to get (log)-probabilities
        norm_factor = tf.reduce_logsumexp(
            joint_likelihood, axis=1, keepdims=True)
        # print("norm_factor:")
        # print(sess0.run(norm_factor))
        log_prob = joint_likelihood - norm_factor
        # exp to get the actual probabilities, ensure prob sum of each class is 1
        return tf.exp(log_prob)

    def get_word_log_cond(self,_word_vec):
        assert self.distrib is not None
        # print(self.dist.scale.shape)
        num_classes, num_features = map(int, self.distrib.scale.shape)
        # print(nb_classes)
        # print(nb_features)

        # Conditional probabilities log P(x|c) with shape
        # (nb_samples, nb_classes)
        data_tiled=tf.tile(_word_vec, [1, num_classes])
        class_and_feature_mat=tf.reshape(data_tiled, [-1, num_classes, num_features])
        log_prob_mat=self.dist.log_prob(class_and_feature_mat)
        #each line represents a data item, sum of log p(x|c)
        condition_log_prob_mat = tf.reduce_sum(log_prob_mat,axis=2)

        return condition_log_prob_mat

    def vec_predict(self,_vec):
        assert self.distrib is not None
        sess=tf.Session()
        saver.restore(sess,self.save_path)
        nb_classes, nb_features = map(int, self.dist.scale.shape)
        tf_tile=tf.tile(_vec, [1, nb_classes])
    
    def get_vec_log_cond_prob(self,_vec):
        assert self.distrib is not None
        # vec_tile=tf.tile(_vec,[self.class_num])
        # vec_reshape=tf.reshape(vec_tile,[self.class_num,self.dim_num])
        cond_prob_mat=self.distrib.log_prob(_vec)
        cond_log_prob_list=tf.reduce_sum(cond_prob_mat,axis=1)
        sess=self.my_sess
        # sess.run(tf.global_variables_initializer())
        return(sess.run(cond_log_prob_list))

    def sentence_predict(self,_word_vecs,_log_p_class_list,_id_to_rel):
        # assert self.distrib is not None
        assert self.my_sess is not None
        sess=self.my_sess
        word_num=len(_word_vecs)
        
        # vec_0=_word_vecs[0]
        # cond_log_cond_prob_0=self.get_vec_log_cond_prob(vec_0)
        # print(cond_log_cond_prob_0)

        # sentence_log_prob=_log_p_class_list
        sentence_log_prob=np.zeros([self.class_num])
        for word_vec in _word_vecs:
            word_log_cond_prob=self.get_vec_log_cond_prob(word_vec)
            sentence_log_prob+=word_log_cond_prob

        # print(sentence_log_prob)
        rel_id=find_max_index(sentence_log_prob)
        sub_rel_name=_id_to_rel[rel_id]
        return sub_rel_name
            


        

    def restore_distrib(self):
        assert self.class_num is not None
        assert self.dim_num is not None
        
        sess=tf.Session()

        self.test_variable=tf.Variable(initial_value=tf.zeros([2,2]))
        self.model_mean=tf.Variable(initial_value=tf.zeros([self.class_num,self.dim_num]))
        self.model_var=tf.Variable(initial_value=tf.zeros([self.class_num,self.dim_num]))
        
        saver=tf.train.Saver({"model_mean":self.model_mean,"model_var":self.model_var})
        sess.run(tf.global_variables_initializer())
        sess.run(self.model_mean)
        sess.run(self.model_var)
        saver.restore(sess,self.var_save_path)
        sess.run(self.model_mean)
        sess.run(self.model_var)

        self.distrib = tf.distributions.Normal(loc=self.model_mean, scale=tf.sqrt(self.model_var))

        self.my_sess=sess




        


def instance2dict(instance):
	# convert the instance into a dict
	tmp_dict = {}
	tmp_dict['question'] = instance[0].split('\t')[1]
	tmp_dict['logical_form'] = instance[1].split('\t')[1]
	tmp_dict['parameters'] = instance[2].split('\t')[1]
	tmp_dict['question_type'] = instance[3].split('\t')[1]
	return tmp_dict

def test_instance2dict(instance):
    tmp_dict = {}
    tmp_dict['question'] = instance[0].split('\t')[1]
    # tmp_dict['logical_form'] = instance[1].split('\t')[1]
    tmp_dict['parameters'] = instance[2].split('\t')[1]
    tmp_dict['question_type'] = instance[3].split('\t')[1]
    return tmp_dict


def fetch_instances(file):
	# fetch data from the original file
	lines = file.readlines()
	instance_list = []
	tmp_instance = []
	for line in lines:
		if line.strip()=="==================================================":
			instance_list.append(instance2dict(tmp_instance))
			tmp_instance = []
		else:
			tmp_instance.append(line)
	return instance_list

def test_fetch_instances(file):
	# fetch data from the original file
	lines = file.readlines()
	instance_list = []
	tmp_instance = []
	for line in lines:
		if line.strip()=="==================================================":
			instance_list.append(test_instance2dict(tmp_instance))
			tmp_instance = []
		else:
			tmp_instance.append(line)
	return instance_list

def get_words_from_instance(_instance):
    qwords=["who","when","what","where","how","which","why","whom","whose"]
    be_verbs=["was","were","is","am","are"]
    preps=["in","on","for","of"]
    sentence=_instance["question"]
    words_prev=sentence.split(" ")
    words=[]
    for word in words_prev:
        word=word.strip()
        word=word_rm_punc(word).strip()
        if((word in qwords) or(word in be_verbs)or(word in preps)):
            continue
        if word is None:
            continue
        if len(word)<1:
            continue
        words.append(word)
    return words

def get_all_words_from_instance(_instance):  
    qwords=["who","when","what","where","how","which","why","whom","whose"]
    be_verbs=["was","were","is","am","are"]
    preps=["in","on","for","of"]
    sentence=_instance["question"]
    words_prev=sentence.split(" ")
    words=[]
    for word in words_prev:
        word=word.strip()
        word=word_rm_punc(word)
        if word is not None:
            words.append(word)      
    return words    

#filter, remove punctuation connected to word
def word_rm_punc(_word):
    word=_word.lower()
    # punctuations=[",",".",":","?"]
    prev_len=len(word)
    # valid_len=-1
    # for i in range(prev_len):
    #     tmp_char=word[i]
    #     if(tmp_char<'a' or tmp_char>'z'):
    #         valid_len=i
    #         break
    # if(valid_len==-1):
    #     return word
    # elif(valid_len==0):
    #     return None
    # else:
    #     return word[:valid_len]
    last_char=word[prev_len-1]
    if(last_char<'a' or last_char>'z'):
        return word[:prev_len-1]
    else:
        return word
    

def get_all_words(_train_path,_dev_path,_test_path):
    # all words, no duplicate
    words_set=set()
    train_file=open(_train_path,'r',encoding="utf-8")
    dev_file=open(_dev_path,'r',encoding="utf-8")
    test_file=open(_test_path,'r',encoding="utf-8")
    train_instances=fetch_instances(train_file)
    dev_instances=fetch_instances(dev_file)
    test_instances=test_fetch_instances(test_file)

    for instance in train_instances:
        instance_words=get_all_words_from_instance(instance)
        for word in instance_words:
            word=word_rm_punc(word)
            if (word==None):
                continue
            else:
                words_set.add(word)
    for instance in dev_instances:
        instance_words=get_all_words_from_instance(instance)
        for word in instance_words:
            word=word_rm_punc(word)
            if (word==None):
                continue
            else:
                words_set.add(word)
    for instance in test_instances:
        instance_words=get_all_words_from_instance(instance)
        for word in instance_words:
            word=word_rm_punc(word)
            if (word==None):
                continue
            else:
                words_set.add(word)

    train_file.close()
    dev_file.close()
    test_file.close()

    return list(words_set)

# need to eliminate punctuations
def save_all_words(_train_path,_dev_path,_test_path,_words_path):
    words_no_dupl=get_all_words(_train_path,_dev_path,_test_path)
    words_path=_words_path
    words_file=open(words_path,"w",encoding="utf-8")
    for word in words_no_dupl:
        words_file.write(word+"\n")
    words_file.close()

def save_word_vec(_words_path,_glove_path,_my_word_vec_path):
    words_file=open(_words_path,"r",encoding="utf-8")
    glove_file=open(_glove_path,"r",encoding="utf-8")
    my_word_vec_file=open(_my_word_vec_path,"w",encoding="utf-8")
    glove_lines=glove_file.readlines()
    glove_dim=300
    glove_dict={}
    # line_cnt=0
    for glove_line in glove_lines:
        line_items=glove_line.strip().split(" ",1)
        glove_word=line_items[0].strip()
        glove_vec_str=line_items[1].strip()
        glove_dict[glove_word]=glove_vec_str
        # line_cnt+=1
        # if(line_cnt%10000==0):
        #     print(glove_vec_str+"\n")
    
    print("read glove finish\n")
    
    my_dict={}
    words_lines=words_file.readlines()
    for word_line in words_lines:
        word=word_line.strip()
        if(word in glove_dict):
            my_dict[word]=glove_dict[word]
    print("my_dict construction finish\n")
    
    for word,vec in my_dict.items():
        my_word_vec_file.write(word)
        my_word_vec_file.write(" "+vec+"\n")


    words_file.close()
    glove_file.close()
    my_word_vec_file.close()

def restore_my_word_vec(_my_word_vec_path,_vec_dim):
    my_word_vec_file=open(_my_word_vec_path,"r",encoding="utf-8")
    word_to_vec={}
    lines=my_word_vec_file.readlines()
    for line in lines:
        line_items=line.strip().split(" ")
        word=line_items[0].strip()
        vec=[float(line_items[i]) for i in range(1,_vec_dim+1)]
        word_to_vec[word]=vec
    return word_to_vec


def get_naive_bayes_model(_sub_index,_train_path,_word_to_vec,_var_save_prefix,_dim_num,
_class_num):
    print("get naive bayes model for sub relation "+str(_sub_index))
    naive_bayes_classifier=Naive_Bayes_Classifier(_sub_index,_var_save_prefix,_dim_num,_class_num)
    sub_rel_list_path="../sub_relations/sub"+str(_sub_index)+".txt"
    sub_rel_list_file=open(sub_rel_list_path,"r",encoding='utf-8')
    lines=sub_rel_list_file.readlines()
    sub_rel_num=len(lines)
    sub_rel_to_id={}
    id_to_sub_rel={}
    for rel_id in range(sub_rel_num):
        sub_rel=lines[rel_id].strip()
        sub_rel_to_id[sub_rel]=rel_id
        id_to_sub_rel[rel_id]=sub_rel
    sub_rel_list_file.close()

    train_data_file=open(_train_path,'r',encoding='utf-8')
    instance_list=fetch_instances(train_data_file)
    # relation id of this word
    #label
    words_sub_rel=[]
    # feature
    words_vec=[]
    for instance in instance_list:
        if((instance['question_type'].strip())!="single-relation"):
            continue
        else:
            # print(instance)
            logical_items=instance['logical_form'].split(' ')
            mso_item=logical_items[4].strip()
            tmp_list=mso_item.split(":")
            sub_relation_items=tmp_list[1].split(".")
            sub_rel=sub_relation_items[_sub_index]
            sub_rel_id=sub_rel_to_id[sub_rel]

            # get words from this distance
            words=get_words_from_instance(instance)
            # find word vec
            for word in words:
                word=word_rm_punc(word)
                if word in _word_to_vec:
                    words_vec.append(_word_to_vec[word])
                    words_sub_rel.append(sub_rel_id)
    # print("words_vec[0]: ")
    # print(words_vec[0])
    # print("words_sub_rel: ")
    # print(words_sub_rel)
    # print(len(words_vec))
    # print(len(words_sub_rel))
    
    train_data_file.close()
    print("fit: ")
    naive_bayes_classifier.fit(words_vec,words_sub_rel)
    return naive_bayes_classifier
    

def get_sub_rel_dict(_sub_rel_path):
    sub_rel_file=open(_sub_rel_path,'r',encoding='utf-8')
    rel_to_id={}
    id_to_rel={}
    lines=sub_rel_file.readlines()
    tmp_id=0
    for line in lines:
        rel_word=line.strip()
        id_to_rel[tmp_id]=rel_word
        rel_to_id[rel_word]=tmp_id
        tmp_id+=1
    sub_rel_file.close()
    return rel_to_id,id_to_rel

def count_sub_rel(_train_path,_rel_pos,_rel_to_id):
    train_file=open(_train_path,'r',encoding='utf-8')
    train_instance_list=fetch_instances(train_file)
    train_file.close()
    sub_rel_num=len(_rel_to_id)
    sub_rel_count_list=np.zeros([sub_rel_num])
    
    for instance in train_instance_list:
        if((instance['question_type'].strip())!="single-relation"):
            continue
        else:
            # print(instance)
            logical_items=instance['logical_form'].split(' ')
            mso_item=logical_items[4].strip()
            tmp_list=mso_item.split(":")
            sub_relation_items=tmp_list[1].split(".")
            sub_rel=sub_relation_items[_rel_pos].strip()
            rel_id=_rel_to_id[sub_rel]
            sub_rel_count_list[rel_id]+=1
    return sub_rel_count_list

def find_max_index(_list):
    max_val=-INF
    max_index=-1
    list_len=len(_list)
    for i in range(list_len):
        tmp_val=_list[i]
        if(tmp_val>max_val):
            max_val=tmp_val
            max_index=i
    return max_index



def main(argv):
    tf.reset_default_graph()
    train_path="../dataset/EMNLP.train"
    dev_path="../dataset/EMNLP.dev"
    dev_res_path="../dataset/EMNLP_dev_res.txt"
    test_path="../dataset/EMNLP.test"
    test_res_path="../dataset/EMNLP_test_res.txt"
    glove_path="../glove.txt"
    words_path="words.txt"
    my_word_vec_path="my_words_vec.txt"
    var_save_folder="distributions/"
    var_save_prefix=var_save_folder+"sub"
    vec_dim=300

    # correspond to 3 sub relation
    class_num_list=[]
    sub_relations_log_probs=[]
    rel_to_id_list=[]
    id_to_rel_list=[]

    for sub_rel_index in range(3):
        sub_rel_path="../sub_relations/sub"+str(sub_rel_index)+".txt"
        rel_to_id,id_to_rel=get_sub_rel_dict(sub_rel_path)
        rel_to_id_list.append(rel_to_id)
        id_to_rel_list.append(id_to_rel)
        class_num=len(rel_to_id)
        class_num_list.append(class_num)
        sub_rel_count_list=count_sub_rel(train_path,sub_rel_index,rel_to_id)
        sub_rel_count_sum=np.sum(sub_rel_count_list)
        sub_rel_prob_list=sub_rel_count_list/sub_rel_count_sum
        sub_rel_log_prob_list=np.log(sub_rel_prob_list)
        sub_relations_log_probs.append(sub_rel_log_prob_list)

    my_word_to_vec=restore_my_word_vec(my_word_vec_path,vec_dim)

    # train naive bayes classifier and store the model
    if_train_list=[False,False,False]
    for classifier_id in range(3):
        if_train=if_train_list[classifier_id]
        if(if_train==False):
            continue
        # train this classifier
        class_num=class_num_list[classifier_id]
        classifier=get_naive_bayes_model(classifier_id,train_path,my_word_to_vec,var_save_prefix,
        vec_dim,class_num)

    # restore model to predict
    naive_bayes_classifiers=[]
    for classifier_id in range(3):
        class_num=class_num_list[classifier_id]
        naive_bayes_classifier=Naive_Bayes_Classifier(classifier_id,var_save_prefix,vec_dim,
        class_num)
        naive_bayes_classifier.restore_distrib()
        naive_bayes_classifiers.append(naive_bayes_classifier)
    # naive_bayes_classifier_0=Naive_Bayes_Classifier(0,var_save_prefix,vec_dim,class_num_0)
    # naive_bayes_classifier_0.restore_distrib()



    # predict test file
    test_file=open(test_path,"r",encoding="utf-8")
    test_instances=test_fetch_instances(test_file)
    test_file.close()
    test_res_file=open(test_res_path,'w',encoding="utf-8")

    test_file=open(test_path,"r",encoding="utf-8")
    raw_test_lines=test_file.readlines()
    raw_test_instances=[]
    raw_test_line_num=len(raw_test_lines)
    print("line num of test file: "+str(raw_test_line_num))
    raw_test_instance_num=int(raw_test_line_num/5)
    for instance_id in range(raw_test_instance_num):
        raw_instance=[]
        for i in range(4):
            raw_instance.append(raw_test_lines[5*instance_id+i].strip())
        raw_test_instances.append(raw_instance)
    
    instance_id=0
    for instance in test_instances:
        # print("new loop\n")
        if(instance["question_type"].strip()!="single-relation"):
            instance_id+=1
            continue
        print("instance id: "+str(instance_id))
        words=get_words_from_instance(instance)
        print(words)
        words_vecs=[]
        for word in words:
            if word not in my_word_to_vec:
                continue
            word_vec=my_word_to_vec[word]
            words_vecs.append(word_vec)
        prediction_list=[]
        for classifier_id in range(3):
            classifier=naive_bayes_classifiers[classifier_id]
            sub_rel_log_prob=sub_relations_log_probs[classifier_id]
            id_to_rel=id_to_rel_list[classifier_id]
            prediction=classifier.sentence_predict(words_vecs,sub_rel_log_prob,id_to_rel)
            prediction_list.append(prediction)
        
        print("prediction: ")
        print(prediction_list)

        raw_test_instance=raw_test_instances[instance_id]
        raw_question_line=raw_test_instance[0]
        raw_parameter_line=raw_test_instance[2]
        raw_type_line=raw_test_instance[3]

        # dev_res_file.write("<question id="+str(instance_id)+">\t")
        # dev_res_file.write(instance["question"].strip()+"\n")
        test_res_file.write(raw_question_line+"\n")
        test_res_file.write("<logical form id="+str(instance_id)+">\t")
        relation=prediction_list[0]+"."+prediction_list[1]+"."+prediction_list[2]
        test_res_file.write("( lambda ?x ( mso:"+relation+" ")
        parameter_items=instance["parameters"].split(" ")
        entity=parameter_items[0].strip()
        test_res_file.write(entity)
        test_res_file.write(" ?x ) )\n")
        test_res_file.write(raw_parameter_line+"\n")
        test_res_file.write(raw_type_line+"\n")
        test_res_file.write("==================================================\n")
        
        instance_id+=1

    test_res_file.close()


    # # predict develop file
    # dev_file=open(dev_path,"r",encoding="utf-8")
    # dev_instances=fetch_instances(dev_file)
    # dev_file.close()
    # dev_res_file=open(dev_res_path,'w',encoding="utf-8")

    # dev_file=open(dev_path,"r",encoding="utf-8")
    # raw_dev_lines=dev_file.readlines()
    # raw_dev_instances=[]
    # raw_dev_line_num=len(raw_dev_lines)
    # print("line num of dev file: "+str(raw_dev_line_num))
    # raw_dev_instance_num=int(raw_dev_line_num/5)
    # for instance_id in range(raw_dev_instance_num):
    #     raw_instance=[]
    #     for i in range(4):
    #         raw_instance.append(raw_dev_lines[5*instance_id+i].strip())
    #     raw_dev_instances.append(raw_instance)
    
    # dev_instance_cnt=0
    # instance_id=0
    # for instance in dev_instances:
    #     # print("new loop\n")
    #     if(instance["question_type"].strip()!="single-relation"):
    #         instance_id+=1
    #         continue
    #     # if(dev_instance_cnt>0):
    #     #     break
    #     # dev_instance_cnt+=1
    #     words=get_words_from_instance(instance)
    #     print(words)
    #     words_vecs=[]
    #     for word in words:
    #         if word not in my_word_to_vec:
    #             continue
    #         word_vec=my_word_to_vec[word]
    #         words_vecs.append(word_vec)
    #     prediction_list=[]
    #     for classifier_id in range(3):
    #         classifier=naive_bayes_classifiers[classifier_id]
    #         sub_rel_log_prob=sub_relations_log_probs[classifier_id]
    #         id_to_rel=id_to_rel_list[classifier_id]
    #         prediction=classifier.sentence_predict(words_vecs,sub_rel_log_prob,id_to_rel)
    #         prediction_list.append(prediction)
        
    #     print("prediction: ")
    #     print(prediction_list)

    #     raw_dev_instance=raw_dev_instances[instance_id]
    #     raw_question_line=raw_dev_instance[0]
    #     raw_parameter_line=raw_dev_instance[2]
    #     raw_type_line=raw_dev_instance[3]

    #     # dev_res_file.write("<question id="+str(instance_id)+">\t")
    #     # dev_res_file.write(instance["question"].strip()+"\n")
    #     dev_res_file.write(raw_question_line+"\n")
    #     dev_res_file.write("<logical form id="+str(instance_id)+">\t")
    #     relation=prediction_list[0]+"."+prediction_list[1]+"."+prediction_list[2]
    #     dev_res_file.write("( lambda ?x ( mso:"+relation+" ")
    #     parameter_items=instance["parameters"].split(" ")
    #     entity=parameter_items[0].strip()
    #     dev_res_file.write(entity)
    #     dev_res_file.write(" ?x ) )\n")
    #     dev_res_file.write(raw_parameter_line+"\n")
    #     dev_res_file.write(raw_type_line+"\n")
    #     dev_res_file.write("==================================================\n")
        
    #     instance_id+=1

    # dev_res_file.close()
        


    

    





if __name__=="__main__":
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)