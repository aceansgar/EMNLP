def instance2dict(instance):
	# convert the instance into a dict
	tmp_dict = {}
	tmp_dict['question'] = instance[0].split('\t')[1]
	tmp_dict['logical_form'] = instance[1].split('\t')[1]
	tmp_dict['parameters'] = instance[2].split('\t')[1]
	tmp_dict['question_type'] = instance[3].split('\t')[1]
	return tmp_dict

def fetch_data(file):
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

def write_sub_file(sub_file,sub_set):
    for sub_rel in sub_set:
        sub_file.write(sub_rel+"\n")


sub_relation_folder="../sub_relations/"
train_data_path="../dataset/EMNLP.train"

train_data_file=open(train_data_path,'r',encoding='utf-8')
sub_pathes=[]
for i in range(3):
    sub_pathes.append(sub_relation_folder+"sub"+str(i)+".txt")
sub_files=[]
for i in range(3):
    tmp_sub_file=open(sub_pathes[i],'w',encoding='utf-8')
    sub_files.append(tmp_sub_file)

sub_sets=[]
for i in range(3):
    sub_sets.append(set())

instance_list=fetch_data(train_data_file)
for instance in instance_list:
    if((instance['question_type'].strip())!="single-relation"):
        continue
    else:
        # print(instance)
        logical_items=instance['logical_form'].split(' ')
        mso_item=logical_items[4].strip()
        tmp_list=mso_item.split(":")
        sub_relation_items=tmp_list[1].split(".")
        # print(sub_relation_items)
        for i in range(3):
            sub_sets[i].add(sub_relation_items[i])

for i in range(3):
    write_sub_file(sub_files[i],sub_sets[i])
