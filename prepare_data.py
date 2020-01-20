import torch
import ast


class Prepare:
	def __init__(self):
		self.path_text_data="text.txt"
		self.folder_embed="embed/"
		self.name_embed_data_file="embed_data_file.txt"
		self.name_text_data_file="text_data_file.txt"
	def read_text_data(self):
		data=[]
		with open(self.path_text_data,"r") as f:
			data = f.readlines()
		return data
	def show_data(self):
		data=self.read_text_data()
		print(data[0])
	def prepare_text_data(self):
		data=self.read_text_data()
		fresh_data=[]
		path_data=[]
		for i in range(len(data)):
			path,text=data[i].split(" ",1)
			fresh_data.append(text)
			path_data.append(path)
		return path_data,fresh_data
	def show_prepare_data_text(self):
		path,data=self.prepare_text_data()
		print(path[0],data[0])
	def read_embed_data(self):
		embed_huge_data=[]
		path_data,text_data=self.prepare_text_data()
		for i in range(len(path_data)):
			tmp_path=path_data[i]
			tmp_path=self.folder_embed+tmp_path+".flac.pt"
			tmp_data=torch.load(tmp_path)
			embed_huge_data.append(tmp_data)
		dictionaty_data_path_embed={"embed":embed_huge_data,"path":path_data,"text":text_data}
		return dictionaty_data_path_embed
	def show_embed(self):
		embed=self.read_embed_data()
		embed=embed["embed"]
		print(embed[1]) 
	def prepare_embed_data(self):
		embed_data=self.read_embed_data()
		embed=embed_data["embed"]
		path_data=embed_data["path"]
		text_data=embed_data["text"]
		fresh_embed_list=[]
		huge_second_list_embed=[]
		for i in range(len(embed)):
			tmp_embed=embed[i]
			tmp_embed=tmp_embed.tolist()
			tmp_list_embed=[]
			second_tmp_list_embed=[]
			for j in range(len(tmp_embed)):
				tmp_elemen_embed=tmp_embed[j]
				tmp_elemen_embed=str(tmp_elemen_embed)
				tmp_elemen_embed=tmp_elemen_embed[0:5]
				tmp_elemen_embed=float(tmp_elemen_embed)
				tmp_elemen_embed=1000*tmp_elemen_embed
				tmp_elemen_embed=int(tmp_elemen_embed)
				if tmp_elemen_embed < 10:
					tmp_elemen_embed = 0
				tmp_list_embed.append(tmp_elemen_embed)
			fresh_embed_list.append(tmp_list_embed)
			for k in range(0,len(tmp_list_embed)-3):
				if tmp_list_embed[k] == 0 and tmp_list_embed[k+1]==0 and  tmp_list_embed[k+2]==0:
					tmp_list_embed[k]=None
					tmp_list_embed[k+1]=None
					tmp_list_embed[k+2]=None
					second_tmp_list_embed.append("|")
				elif tmp_list_embed[k] != None:
					second_tmp_list_embed.append(tmp_list_embed[k])
			huge_second_list_embed.append(second_tmp_list_embed)
		main_dictionary={"embed":huge_second_list_embed,"text":text_data,"path":path_data}
		return main_dictionary

	def replace_180(self,list_embed):
		new_list=[]
		for i in range(len(list_embed)):
			if list_embed[i] == "|":
				new_list.append("|")
			elif int(list_embed[i]) > 179:
				new_list.append(179)
			else:
				new_list.append(list_embed[i])
		return new_list
	def show_prepare_embed(self):
		data=self.prepare_embed_data()
		data=data["embed"]
		print(data[1])
	def create_china(self,alpha):
		test_list = [] 
		for i in range(0, 190): 
		    test_list.append(alpha) 
		    alpha = chr(ord(alpha) + 1)  
		return test_list
	def replace_liter_embed(self,list_num,dictionary):
		tmp_list=[]
		number,alha=dictionary["nums"],dictionary["alp"]
		for i in range(len(list_num)):
			if list_num[i] == "|":
				tmp_list.append("|")
			else:
				temp=number.index(list_num[i])
				temp_chpter=alha[temp]
				tmp_list.append(temp_chpter)
		return tmp_list

	def create_data_file(self):
		data=self.prepare_embed_data()
		embed=data["embed"]
		text=data["text"]
		
		new_embed=[]
		for i in range(len(embed)):
			temp_embed=self.replace_180(embed[i])
			new_embed.append(temp_embed)
		embed = new_embed

		china = self.create_china("诶")
		list_of_numer=[i for i in range(190)]
		dictionary={"nums":list_of_numer,"alp":china}

		new_embed=[]

		for i in range(len(embed)):
			tmp_embed=self.replace_liter_embed(embed[i],dictionary)
			new_embed.append(tmp_embed)
		embed=new_embed
		#create embed file : 
		with open(self.name_embed_data_file,"w") as f :
			for i in range(len(embed)):
				tmp_line=str(embed[i])
				f.write(tmp_line)
				f.write("\n")
		# create text file : 
		with open(self.name_text_data_file,"w") as f :
			for i in range(len(text)):
				tmp_line=str(text[i])
				f.write(tmp_line)

prepare=Prepare()
prepare.create_data_file()



'''
[109, '|', 21, 43, '|', 67, 20, 0, 14, 70, 139, 96, '|', 65, '|', 128, 0, 11, '|', 36, '|', 22, 22, 0, 85, 40, 
'|', 0, 140, '|', 0, 11, 0, 50, '|', 0, 109, 139, 0, 17, 22, 0, 66, 61, 32, 0, 136, 37, '|', 64, 0, 27, '|', 
0, 85, '|', 133, '|', 0, 20, 0, 66, '|', '|', '|', '|', 54, 105, 0, 150, 13, 87, 0, 165, '|', 89, 26, '|', 25, 
59, 67, 59, 0, 56, 129, 118, 0, 25, 84, 0, 52, 0, 28, 97, 150, 98, 66, 0, 24, '|', 259, 111, '|', 0, 23, 30, 
51, 65, '|', 78, '|', 13, 79, 0, 34, 34, 162, 36, 166, 0, 154, 15, '|', 100, 0, 38, 188, 110, '|', 29, '|', '|', 
'|', '|', 0, 86, 168, 72, 0, 148, 41, 19, '|', '|', 0, 90, '|', 80, '|', 34, 26, 0, 35, '|', 0, 84, 0, 28, '|', 
'|', 0, 29, 67, 3614, 127, 78, 70, 0, 55, 252, '|', 0, 92, 156, 0, 63, 59, 0]


MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL

['豣', '|', '谷', '谋', '谡', '|', '诶', '谹', '谊', '诶', '谄', '谼', '貁', '豖', '|', '豏', '谷', '|', '诶', '豶', '诶',
'谁', '|', '诶', '谚', '|', '豱', '谌', '谌', '诶', '豋', '谞', '|', '豇', '诶', '貂', '|', '谀', '诶', '谁', '诶', '谨',
'|', '|', '豣', '貁', '诶', '谇', '谌', '诶', '谸', '谳', '谖', '诶', '豾', '谛', '|', '诶', '谶', '诶', '谑', '|', '|',
'豋', '|', '诶', '豻', '|', '|', '谊', '诶', '谸', '|', '|', '|', '|', '|', '|', '谬', '豟', '诶', '貌', '调', '豍',
'诶', '貛', '|', '诶', '豏', '谐', '|', '豃', '谏', '谱', '谹', '谱', '诶', '谮', '豷', '豬', '诶', '谏', '豊', '诶',
'谪', '诶', '谒', '豗', '貌', '豘', '谸', '诶', '谎', '|', '谳', '販', '豥', '|', '貐', '诶', '谍', '谔', '谩', '谷', 
'|', '诶', '豄', '|', '豞', '调', '豅', '诶', '谘', '谘', '貘', '谚', '貜', '诶', '貐', '谅', '|', '貐', '豚', '诶', 
'谜', '販', '豤', '|', '谍', '谓', '|', '谩', '|', '谋', '|', '|', '诶', '豅', '诶', '豌', '貞', '谾', '诶', '貊', 
'谟', '谉', '|', '|', '诶', '豆', '诶', '豐', '|', '豫', '豆', '|', '诶', '谘', '谐', '诶', '谙', '|', '|', '豊', 
'诶', '谒', '|', '|', '|', '诶', '谓', '谹', '販', '豵', '豄', '谼', '诶', '谭', '販', '|', '|', '豒', '貒', '诶',
 '谵', '谱', '诶']

['豣', '|', '谋', '谡', '诶', '|', '谊', '诶', '谄', '谼', '貁', '豖', '|', '谷', '诶', '|', '诶', '谁', '诶', '|', '|',
'谌', '谌', '诶', '豋', '谞', '|', '诶', '貂', '|', '诶', '谁', '诶', '谨', '诶', '诶', '|', '貁', '诶', '谇', '谌', '诶', 
'谸', '谳', '谖', '诶', '豾', '谛', '诶', '|', '诶', '谑', '诶', '诶', '|', '诶', '|', '诶', '诶', '|', '诶', '谸', '诶', 
'诶', '诶', '诶', '诶', '诶', '诶', '诶', '诶', '诶', '|', '豟', '诶', '貌', '调', '豍', '诶', '貛', '诶', '|', '谐', '|', 
'谏', '谱', '谹', '谱', '诶', '谮', '豷', '豬', '诶', '谏', '豊', '诶', '谪', '诶', '谒', '豗', '貌', '豘', '谸', '诶', '谎',
'|', '販', '豥', '|', '诶', '谍', '谔', '谩', '谷', '诶', '|', '|', '调', '豅', '诶', '谘', '谘', '貘', '谚', '貜', '诶', 
'貐', '谅', '|', '豚', '诶', '谜', '販', '豤', '|', '谓', '|', '|', '诶', '诶', '诶', '|', '诶', '豌', '貞', '谾', '诶', 
'貊', '谟', '谉', '诶', '诶', '诶', '|', '诶', '豐', '|', '豆', '诶', '|', '谐', '诶', '谙', '诶', '诶', '|', '诶', '谒', 
'诶', '诶', '诶', '诶', '诶', '|', '谹', '販', '豵', '豄', '谼', '诶', '谭', '販', '诶', '诶', '|', '貒', '诶', '谵', '谱']


'''
