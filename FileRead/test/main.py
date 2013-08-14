from FileRead import readASCIIintoArray

def main():
    filenm = "filereadtest.dat"
    outdata = readASCIIintoArray(filenm, 1,delim='  ')
    print type(outdata)
    print outdata
    
if __name__ == "__main__":
    main()
    