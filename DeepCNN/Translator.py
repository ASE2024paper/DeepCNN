def readImport(fileName):
    file_path = "{0}.txt".format(fileName)

    try:
        with open(file_path, "r") as file:
            python_code = file.read()
            return python_code
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print("An error occurred:", str(e))


file_path = "model.py"

try:
    with open(file_path, "r") as file:
        python_code = file.read()

        # Find the index of "Sequential()"
        sequential_index = python_code.find("Sequential()")

        if sequential_index != -1:
            part1 = python_code[:sequential_index]
            part2 = python_code[sequential_index:]
            
            print(readImport("import"))
            print("Part 1 (Before 'Sequential()'):")
            print(part1)
            
            print(readImport("finish"))

            print("\nPart 2 (After 'Sequential()'):")
            print(part2)
        else:
            print("'Sequential()' not found in the code.")
except FileNotFoundError:
    print("The specified file was not found.")
except Exception as e:
    print("An error occurred:", str(e))
