import os
from computer_python.Log import tagFormat, producerFormat, prettyPrint, timeFormat, messageFormat

path = os.path.join(os.getcwd(), "logs")

def getFiles(directory:str) -> list[str]:
  dir = os.listdir(directory)
  files:list[str] = []
  files = [fileName for fileName in dir   if os.path.isfile(os.path.join(directory, fileName)  )]
  files = [fileName for fileName in files if fileName.casefold().endswith  ('.txt'.casefold()  )]
  files = [fileName for fileName in files if fileName.casefold().startswith('log.'.casefold()  )]
  files = [fileName for fileName in files if "nonviewable".casefold() not in os.path.join(directory, fileName).casefold()]
  for fileName in dir:
    if os.path.isdir(os.path.join(directory, fileName)):
      files += getFiles(os.path.join(directory, fileName))
  return files

def filterLines(fileName:str, tags:list[str] | None = None, producers:list[str] | None  = None,time: tuple[float,float] | None = None, messages:list[str] | None  = None):
  """
  Filters the file contents based on the given filters, general whitelist.
  :param fileName: The name of the file to filter, general whitelist.
  :param tags: A list of tags to filter by, general whitelist.
  :param producers: A list of producers to filter by, general whitelist.
  :param messages: A list of messages to filter by, secondary whitelis.
  :param time: A tuple of start and end time to filter by, secondary whitelist.
  :return: A list of lines that match the filters.
  
  If no tags or producers are given, all lines will be provided to secondary filters.
  If a secondary filter is empty or None, it will not be applied.
  if there are tags or producers, the lines will be filtered by those tags and producers then by message content. 
  this means that either tags and producers must be present in the line and the message must match and time must match.
  """
  if not os.path.isfile(fileName) or not os.access(fileName, os.R_OK):
    print(f"File {fileName} does not exist.")
    return []
  
  out:list[str] = []
  
  possibilities = []
  for tag in tags if tags is not None else []:
    possibilities.append(tagFormat(tag))
  for producer in producers if producers is not None else []:
    possibilities.append(producerFormat(producer))
  
  with open(fileName, 'r') as file:
    if(len(possibilities) != 0):
      lines = file.readlines()
      for line in lines:
        if any(possibility in line for possibility in possibilities):
          out.append(line.strip())
    else:
      out = [line.strip() for line in file.readlines()]
  
  #sort the output again by message
  fails:list[str] = []
  for line in out:
    if messages is not None:
      striped_line = line.split("(")[1].split(")")[0] if "(" in line and ")" in line else ""
      if not any(message in striped_line for message in messages):
        fails.append(line)
  for fail in fails:
    out.remove(fail)
  fails.clear()
  
  if time is not None:
    start, end = time
    for line in out:
      try:
        time_float = float(line.split("{")[1].split("}")[0]) if "(" in line and ")" in line else -1
        if not (start <= time_float <= end):
          fails.append(line)
      except IndexError:
        print(f"Error parsing time from line: {line}")
        fails.append(line)
  for fail in fails:
    out.remove(fail)
  return out

def printLines(filtered_files:list[tuple[str,list[str]]]) -> None:
  """
  Prints the filtered files and their contents.
  :param directory: The directory to search for files.
  :param tags: A list of tags to filter by, general whitelist.
  :param producers: A list of producers to filter by, general whitelist.
  :param messages: A list of messages to filter by, secondary whitelist.
  :param time: A tuple of start and end time to filter by, secondary whitelist.
  """
  
  for filename, lines in filtered_files:
    print(f"File: {filename}")
    for line in lines:
      prettyPrint(*splitLine(line))

def splitLine(line:str) -> tuple[str, str, str, float]:
  """
  Splits a line into its components.
  :param line: The line to split.
  :return: A tuple of (tag, message, producer, time).
  """
  tag = line.split("[")[1].split("]")[0] if "[" in line and "]" in line else ""
  message = line.split("(")[1].split(")")[0] if "(" in line and ")" in line else ""
  producer = line.split("<")[1].split(">")[0] if "<" in line and ">" in line else ""
  time_str = line.split("{")[1].split("}")[0] if "{" in line and "}" in line else "0.0"
  
  try:
    time_float = float(time_str)
  except ValueError:
    time_float = 0.0
  
  return tag, message, producer, time_float

def generateReport():
  report:list[tuple[str,list[str]]] = []
  filterGroups:list[tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]] = []
  filters:tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None] = (None, None, None, None) # tags, producers, time, messages
  while True:
    command = input("input command, help for help: ")
    command = command.strip().lower()
    if command == "help":
      print("Available commands:")
      print("  filter - add new filter (sub menu)")
      print("  reset - reset all filters and current filtered lines")
      print("  apply - apply the filters and add to current filtered lines")
      print("          be careful, this will not reset the current filtered lines")
      print("          can only be undone by resetting")
      print("          reset filters to prepare for next apply, no matter if apply succeded or not")
      print("  show - show filtered lines")
      print("  save <filename> - save the filtered lines to a file")
      print("                    this will reset the current filtered lines and filters")

      print("  exit - quit the program without saving")
      print("  help - show this help message")
    elif command == "filter":
      filters = filterMenu(filters)
    elif command == "reset":
      filters = (None, None, None, None)
      report = []
      filterGroups = []
      print("Filters reset.")
    elif command == "apply":
      files = getFiles(path)
      if files is not None and len(files) > 0:
        for file in files:
          newLines = filterLines(os.path.join(path,file), filters[0], filters[1], filters[2], filters[3])
          if newLines is not None and len(newLines) > 0:
            report = extendReport(report, (file, newLines))
      else:
        print("No files found with the current filters.")
      filterGroups.append(filters)
      filters = (None, None, None, None)  # Reset filters after applying
    elif command == "show":
      if report is not None and len(report) > 0:
        print("Filtered lines:")
        printLines(report)
      else:
        print("No filtered lines to show. Please apply a filter first.")
    elif command.startswith("save"):
      filename = command.split("save")[1].strip()
      if filename is not None and filename != "":
        file = open(os.path.join(os.getcwd(),filename), "w")
        saveReport(file,filterGroups, report)
        report = []  # Reset report after saving
        filterGroups = []  # Reset filter groups after saving
        filters = (None, None, None, None)  # Reset filters after saving
        print(f"Report saved to {filename}.")
      else:
        print("No filename provided. Please use 'save <filename>'.")
    elif command == "exit":
      print("Exiting without saving.")
      break
    else:
      print("Unknown command. Type 'help' for a list of commands.")
      continue

def saveReport(file, filterGroups:list[tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]], report:list[tuple[str,list[str]]]) -> None:
  """
  Saves the report to a file.
  :param file: The file to save the report to.
  :param report: The report to save.
  """
  cleaned_report = []
  for lines in report:
    cleaned_report = extendReport(cleaned_report, lines)
  file.write("Filtered Report:\n")
  
  #filter information
  file.write       (f"->Filters:\n")
  for i, (tags, producers, time, messages) in enumerate(filterGroups):
    file.write     (f"--->filter{i+1}:\n")
    if tags is not None and len(tags) > 0:
      file.write   (f"----->Tags:\n")
      for tag in tags:
        file.write (f"------->{tag}\n")
    if producers is not None and len(producers) > 0:
      file.write   (f"----->Producer:\n")
      for producer in producers:
        file.write (f"------->{producer}\n")
    if time is not None:
      file.write   (f"----->Time:\n")
      file.write   (f"------->Start: {time[0]:.5f}\n")
      file.write   (f"------->End: {time[1]:.5f}\n")
    if messages is not None and len(messages) > 0:
      file.write   (f"----->Messages:\n")
      for message in messages:
        file.write (f"------->{message}\n")
  
  #lines
  for filename, lines in cleaned_report:
    file.write(f"->File: {filename}\n")
    for line in lines:
      tag, message, producer, time = splitLine(line)
      file.write(f"--->{timeFormat(time)} {producerFormat(producer)} {tagFormat(tag)}: {messageFormat(message)}\n")

def extendReport(report:list[tuple[str,list[str]]], lines:tuple[str,list[str]]) -> list[tuple[str,list[str]]]:
  newFile, newLines = lines
  if newFile not in [f[0] for f in report]:
    report.append((newFile, newLines))
  else:
    for i, (file, existingLines) in enumerate(report):
      if file == newFile:
        report[i] = (file, existingLines + newLines)
        break
  return report

def filterMenu(filters:tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]) -> tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]:
  while True:
    command = input("input command, help for help: ")
    command = command.strip().lower()
    if command == "help":
      print("Available commands:")
      print("  add:")
      print("    add tag <tag> - add a tag to filter by")
      print("    add producer <producer> - add a producer to filter by")
      print("    set time <start> <end> - set a time range to filter by")
      print("    add message <message> - add a message to filter by")
      print("  remove:")
      print("    remove tag <tag> - remove a tag from the filter")
      print("    remove producer <producer> - remove a producer from the filter")
      print("    remove time - remove the time filter")
      print("    remove message <message> - remove a message from the filter")
      print("  show - show the current filters")
      print("  reset - reset all filters")
      print("  done - finish filtering and return to the main menu")
    elif command.startswith("add"):
      command = command.split("add")[1].strip()
      filters = __addFilters(command, filters)
    elif command.startswith("remove"):
      command = command.split("remove")[1].strip()
      filters = __removeFilters(command, filters)
    elif command == "show":
      print("Current filters:")
      print(f"  Tags: {filters[0] if filters[0] is not None else 'None'}")
      print(f"  Producers: {filters[1] if filters[1] is not None else 'None'}")
      print(f"  Time: {filters[2] if filters[2] is not None else 'None'}")
      print(f"  Messages: {filters[3] if filters[3] is not None else 'None'}")
    elif command == "reset":
      filters = (None, None, None, None)
      print("Filters reset.")
    elif command == "done":
      print("Returning to the main menu.")
      break
  return filters  # Placeholder for filter menu implementation

def __addFilters(command:str, filters:tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]) -> tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]:
  if command.startswith("tag"):
    split = command.split(" ")
    tag = " ".join(split[1:]).strip()
    if filters[0] is None:
      filters = ([], filters[1], filters[2], filters[3])
    if filters[0] is not None:
      if tag not in filters[0]:
        filters[0].append(tag)
        print(f"Added tag: {tag}")
      else:
        print(f"Tag {tag} already exists in the filter.")
    else:
      print("failed to add tag, unkown error")
  elif command.startswith("producer"):
    split = command.split(" ")
    producer = " ".join(split[1:]).strip()
    if filters[1] is None:
      filters = (filters[0], [], filters[2], filters[3])
    if filters[1] is not None:
      if producer not in filters[1]:
        filters[1].append(producer)
        print(f"Added producer: {producer}")
      else:
        print(f"Producer {producer} already exists in the filter.")
    else:
      print("failed to add producer, unkown error")
  elif command.startswith("message"):
    split = command.split(" ")
    meassage = " ".join(split[1:]).strip()
    if filters[3] is None:
      filters = (filters[0], filters[1], filters[2], [])
    if filters[3] is not None:
      if meassage not in filters[3]:
        filters[3].append(meassage)
        print(f"Added meassage: {meassage}")
      else:
        print(f"meassage {meassage} already exists in the filter.")
    else:
      print("failed to add meassage, unkown error")
  elif command.startswith("time"):
    try:
      start, end = map(float, " ".join(command.split(" ")[1:]).strip().split(" "))
      filters = (filters[0], filters[1], (start, end), filters[3])
      print(f"Set time filter: {start:.5} to {end:.5}")
    except ValueError:
      print("Invalid time format. Please use 'set time <start> <end>'.")
  return filters

def __removeFilters(command:str, filters:tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]) -> tuple[list[str] | None, list[str] | None, tuple[float, float] | None, list[str] | None]:
  if command.startswith("tag"):
    tag = command.split("tag")[1].strip()
    if filters[0] is None:
      print("No tags to remove.")
      return filters
    if filters[0] is not None:
      if tag in filters[0]:
        filters[0].remove(tag)
        print(f"removed tag: {tag}")
      else:
        print(f"Tag {tag} does not exist in the filter.")
    else:
      print("failed to remove tag, unkown error")
  elif command.startswith("producer"):
    producer = command.split("producer")[1].strip()
    if filters[1] is None:
      print("No producers to remove.")
      return filters
    if filters[1] is not None:
      if producer in filters[1]:
        filters[1].remove(producer)
        print(f"removed producer: {producer}")
      else:
        print(f"Producer {producer} does not exist in the filter.")
    else:
      print("failed to remove producer, unkown error")
  elif command.startswith("meassage"):
    meassage = command.split("meassage")[1].strip()
    if filters[3] is None:
      print("No meassages to remove.")
      return filters
    if filters[3] is not None:
      if meassage in filters[3]:
        filters[3].remove(meassage)
        print(f"removed meassage: {meassage}")
      else:
        print(f"Producer {meassage} does not exist in the filter.")
    else:
      print("failed to remove meassage, unkown error")
  elif command.startswith("time"):
    filters = (filters[0], filters[1], None, filters[3])
  return filters

if __name__ == "__main__":
  print("Log Viewer started.")
  generateReport()
  print("Log Viewer closed.")