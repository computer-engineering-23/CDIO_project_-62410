import time

logFile = None
startTime = 0
noTag = []

def enableLog():
  global logFile, startTime
  tim = time.gmtime()
  logFile = open(f"log.{tim.tm_mday:02d}-{tim.tm_hour:02d}-{tim.tm_min:02d}.txt", "x")
  startTime = time.time()

def blockTag(tag:str):
  """
  Blocks the tag from being printed to the log file.
  """
  global noTag
  noTag.append(tag.casefold())

def printLog(tag:str,message:str,*following):
  global logFile, startTime, noTag
  
  if(tag.casefold() in noTag):
    return
  
  string = f"[{tag.upper()}] " + message
  for i in following:
    string += " " + str(i)
  print(string)
  if(logFile is None):
    return
  timeString:str = f"{"{"}{(time.time() - startTime):3.5f}{"}"} "
  logFile.write(timeString + string + "\n")

def closeLog():
  global logFile
  if(logFile is None):
    print("no open log")
    return
  name = logFile.name
  logFile.close()
  logFile = open(name, "r")
  print(f"closing log with: {len(logFile.readlines())} lines")
  logFile.close()