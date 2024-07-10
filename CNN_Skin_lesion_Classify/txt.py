import os
from os import getcwd

classes=['melanoma','nevus']
sets=['train']

if __name__=='__main__':
    wd=getcwd()
    for se in sets:
        list_file=open('cls_'+ se +'.txt','w')

        datasets_path=se
        print(datasets_path)
        types_name=os.listdir(datasets_path)# os.listdir() method is used to return a list of the names of the files or folders contained in the specified folder
        print(types_name)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id=classes.index(type_name)# Output 0-1
            photos_path=os.path.join(datasets_path,type_name)
            photos_name=os.listdir(photos_path)
            for photo_name in photos_name:
                _,postfix=os.path.splitext(photo_name)#This function is used to separate filenames from extensions
                if postfix not in['.jpg','.png','.jpeg', '.JPG']:
                    continue
                list_file.write(str(cls_id)+';'+'%s/%s'%(wd, os.path.join(photos_path,photo_name)))
                list_file.write('\n')
        list_file.close()


