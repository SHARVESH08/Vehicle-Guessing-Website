import os

root = r"C:\VIT Stuff\SEM-4\FoDS\Project\Dataset\train"

total = 0
for cls in os.listdir(root):
    total += len(os.listdir(os.path.join(root, cls)))

print("Total train images:", total)
