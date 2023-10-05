import os
from fileinput import FileInput
from glob import glob

paths = [y for x in os.walk("keras_nlp") for y in glob(os.path.join(x[0], '*.py'))]


for path in paths:
    with FileInput(files=[path], inplace=True) as file:
        seen_import = False
        for line in file:
            if not seen_import and "import" in line:
                seen_import = True
                print()
            if not seen_import and not line.startswith("#"):
                continue
            print(line, end="")
