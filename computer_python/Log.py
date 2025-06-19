import time

logFile = None
startTime = 0
noTag = []
prettyPrintEnabled = False

def enableLog():
  global logFile, startTime
  tim = time.gmtime()
  logFile = open(f"../logs/log.{tim.tm_mday:02d}-{tim.tm_hour:02d}-{tim.tm_min:02d}.txt", "x")
  startTime = time.time()

def blockTag(tag:str):
  """
  Blocks the tag from being printed to the log file.
  """
  global noTag
  noTag.append(tag.strip().casefold())

def unblockTag(tag:str):
  """
  Unblocks the tag from being printed to the log file.
  """
  global noTag
  if tag.strip().casefold() in noTag:
    noTag.remove(tag.strip().casefold())

def enablePrettyPrint():
  """
  Enables pretty printing of the log messages.
  """
  global prettyPrintEnabled
  prettyPrintEnabled = True

def disablePrettyPrint():
  """
  Disables pretty printing of the log messages.
  """
  global prettyPrintEnabled
  prettyPrintEnabled = False

def printLog(tag:str,message:str,*following,printable:bool=True,producer:str="unkown"):
  global startTime, noTag
  
  if(tag.strip().casefold() in noTag):
    return
  
  for i in following:
    message += " " + str(i)
  
  current_time = time.time() - startTime
  
  if(printable):
    prettyPrint(tag, message, producer, current_time)
  
  logWrite(tag, message, producer,current_time)

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

def tagFormat(tag:str) -> str:
  """
  Formats the tag for printing.
  """
  return f"[{tag.upper().replace('\n', ' ')}]"

def messageFormat(message:str) -> str:
  """
  Formats the message for printing.
  """
  return f"({message.replace('\n', ' ')})"

def producerFormat(producer:str) -> str:
  """
  Formats the producer for printing.
  """
  return f"<{producer.replace('\n', ' ')}>"

def timeFormat(tim:float) -> str:
  """
  Formats the time for printing.
  """
  return f"{tim:.5f}"

def prettyPrint(tag:str,message:str,producer:str,current_time:float):
  """
  Formats the string for printing.
  """
  print(f"{color("time")}{timeFormat(current_time)} {color("producer")}{producerFormat(producer)} {color("tag")}{tagFormat(tag)}:\t{color("message")}{messageFormat(message)}{color()}")

def logWrite(tag:str, message:str, producer:str, current_time:float) -> None:
  """
  Prints the log message to the console and the log file.
  """
  global logFile
  if(logFile is not None):
    logFile.write(f"{timeFormat(current_time)} {producerFormat(producer)} {tagFormat(tag)}:\t{messageFormat(message)}\n")

def color(type:str = "") -> str:
  global prettyPrintEnabled
  if not prettyPrintEnabled:
    return ""  # No color if pretty print is disabled
  colors = {
    "tag": "\033[38;5;4m",  # Blue
    "producer": "\033[38;5;2m",  # Green
    "time": "\033[38;5;5m",  # Magenta
    "message": "\033[38;5;15m",  # white
  }
  return colors.get(type, "\033[0m")  # Default to no color