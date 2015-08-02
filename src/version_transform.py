def transformVersion(file):
    f = open(file, "r")
    fileData = f.read()
    f.close()

    fileData = list(fileData)
    counter = 0
    for char in xrange(len(fileData)):
        if (fileData[char] == "\begin{equation}"): #and (counter % 2 == 0):
            fileData[char] = "\["
            counter += 1
        elif (fileData[char] == "\end{equation}"): #and (counter % 2 != 0):
            fileData[char] = "\]"
            counter += 1

    fileData = "".join(fileData)
    f = open(file + ".new", "w")
    f.write(fileData)
    f.close()
