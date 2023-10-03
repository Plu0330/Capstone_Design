import os
import random

directory_path = r'C:\Users\AIMS Lab\Desktop\Region'

output_file = r'C:\Users\AIMS Lab\Desktop\Region_Shuffle.txt'

txt_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

combined_content = ""
for file_name in txt_files:
    with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
        file_content = file.read()
        combined_content += file_content

content_lines = combined_content.splitlines()
random.shuffle(content_lines)


with open(output_file, 'w', encoding='utf-8') as output:
    for line in content_lines:
        output.write(line + '\n')
