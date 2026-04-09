import os

root = r"C:\VIT Stuff\SEM-4\FoDS\Project\Dataset\train"

classes = os.listdir(root)
print("Total classes:", len(classes))
