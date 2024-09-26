
import os
import random
import string

def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits 
    return ''.join(random.choice(characters) for _ in range(length))


def rename_folders(base_directory='brain_ct/data'):
    for root, dirs, _ in os.walk(base_directory, topdown=False): 
        for folder in dirs:
            original_path = os.path.join(root, folder)
            random_name = generate_random_string()
            new_path = os.path.join(root, random_name)
            os.rename(original_path, new_path)
            print(f'Renamed "{original_path}" to "{new_path}"')


rename_folders()