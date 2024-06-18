from os import listdir

def get_pet_labels(image_dir):
    # Creates list of files in directory
    in_files = listdir(image_dir)
    
    # Creates empty dictionary for the results (pet labels, etc.)
    results_dic = dict()
   
    # Processes through each file in the directory, extracting only the words
    # of the file that contain the pet image label
    for idx in range(0, len(in_files), 1):
       
       # Skips file if starts with . (like .DS_Store of Mac OSX) because it 
       # isn't an pet image file
       if in_files[idx][0] != ".":
           
           # Creates temporary label variable to hold pet label name extracted 
           pet_label = ""

           # Processes each filename to extract the pet labels
           # The filename is split into words using underscores as delimiters
           filename = in_files[idx].lower().split('_')
           
           # Join all the words that are alphabets into a single string
           pet_label = " ".join([word for word in filename if word.isalpha()]).strip()

           # If filename doesn't already exist in dictionary add it and its
           # pet label - otherwise print an error message because indicates 
           # duplicate files (filenames)
           if in_files[idx] not in results_dic:
              results_dic[in_files[idx]] = [pet_label]
           else:
               print("** Warning: Duplicate files exist in directory:", 
                     in_files[idx])
 
    return results_dic
