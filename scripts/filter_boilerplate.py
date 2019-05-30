import sys, re

def proc(filename, outfile):
	print(filename)
	with open(filename, encoding="utf-8") as file:

		out=open(outfile, "w", encoding="utf-8")

		lines=file.readlines()
		start=None
		end=None
		for idx, line in enumerate(lines):


			if re.search("START OF THE PROJECT GUTENBERG", line, re.I) is not None:
				start=idx
			if re.search("START OF THIS PROJECT GUTENBERG", line, re.I) is not None:
				start=idx
			
			# 676, 2364
			if re.search("\*END\*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS\*Ver.04.29.93\*END\*", line, re.I) is not None:
				start=idx

			# 3925
			if re.search("\*END THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS\*Ver.07/27/01\*END\*", line, re.I) is not None:
				start=idx


			# 4721, 4585
			if re.search("tells you about restrictions in how the file may be used.", line, re.I) is not None:
				start=idx

				

			if re.search("End of the Project Gutenberg", line, re.I) is not None:
				end=idx

			if end == None and re.search("End of Project Gutenberg", line, re.I) is not None:
				end=idx

			if end == None and re.search("END OF THE PROJECT GUTENBERG", line, re.I) is not None:
				end=idx

			
			if end == None and re.search("END OF THIS PROJECT GUTENBERG", line, re.I) is not None:
				end=idx

				
		print(start, end)
		data=""
		for i in range(start+1, end-1):
			data+="%s__NEWLINE__" % lines[i].rstrip()

		data=re.sub("^(__NEWLINE__)+", "", data)

		data=re.sub("^Produced by.*?__NEWLINE____NEWLINE__", "", data)
		data=re.sub("^This eBook was produced by.*?__NEWLINE____NEWLINE__", "", data)
		data=re.sub("^E-text prepared by.*?__NEWLINE____NEWLINE__", "", data)
		data=re.sub("^Transcribed from.*?__NEWLINE____NEWLINE__", "", data)
		data=re.sub("__NEWLINE__", "\n", data)
		data=data.lstrip().rstrip()
		out.write(data)
		

		out.close()

		# data=file.read()

		# data=re.sub("\n", "__NEWLINE__", data)


		# data=re.sub("^\*\*\*.*START OF THE PROJECT GUTENBERG.*?\*\*\*", "", data, re.MULTILINE, re.I)
		# data=re.sub("\*\*\*.*?END OF THE PROJECT GUTENBERG.*$", "", data, re.MULTILINE, re.I)

		# data=re.sub("^.*START OF THIS PROJECT GUTENBERG.*?\*\*\*", "", data, re.MULTILINE, re.I)
		# data=re.sub("\*\*\*.*?END OF THIS PROJECT GUTENBERG.*$", "", data, re.MULTILINE, re.I)

		# data=re.sub("END OF the PROJECT GUTENBERG.*$", "", data, re.MULTILINE, re.I)
		# data=re.sub("END OF THIS PROJECT GUTENBERG.*$", "", data, re.MULTILINE, re.I)
		# data=re.sub("END OF PROJECT GUTENBERG.*$", "", data, re.MULTILINE, re.I)

		# data=re.sub("__NEWLINE__", "\n", data)

		# with open(outfile, "w", encoding="utf-8") as out:
		# 	out.write(data)

proc(sys.argv[1], sys.argv[2])