Data and code associated with "The Observed Availability of Data and Code in Earth Science 
and Artificial Intelligence" by Erin A. Jones, Brandon McClung, Hadi Fawad, and Amy McGovern.

For any questions, please contact the corresponding author, Amy McGovern (amcgovern@ou.edu).

Instructions: To reproduce figures, download all associated Python and CSV files and place
              in a single directory.
              Run BAMS_plot.py as you would run Python code on your system.

Code:
BAMS_plot.py: Python code for categorizing data availability statements based on given data
              documented below and creating figures 1-3. 

              Code was originally developed for Python 3.11.7 and run in the Spyder 
              (version 5.4.3) IDE.
              
              Libraries utilized:
                numpy       (version 1.26.4) 
                pandas      (version 2.1.4)
                matplotlib  (version 3.8.0)
              
              For additional documentation, please see code file.

Data:
ASDC_AIES.csv:      CSV file containing relevant availability statement data for Artificial 
                    Intelligence for the Earth Systems (AIES)
ASDC_AI_in_Geo.csv: CSV file containing relevant availability statement data for Artificial 
                    Intelligence in Geosciences (AI in Geo.)
ASDC_AIJ.csv:       CSV file containing relevant availability statement data for Artificial 
                    Intelligence (AIJ)
ASDC_MWR.csv:       CSV file containing relevant availability statement data for Monthly 
                    Weather Review (MWR)


Data documentation:
All CSV files contain the same format of information for each journal. The CSV files above are 
needed for the BAMS_plot.py code attached.

Records were subjectively analyzed by the co-authors above based on the criteria below.

  Records:
	1) Title of paper
		The title of the examined journal article.
	2) Article DOI (or URL)
		A link to the examined journal article. For AIES, AI in Geo., MWR, the DOI is 
		generally given. For AIJ, the URL is given.
	3) Journal name
		The name of the journal where the examined article is published. Either a full
		journal name (e.g., Monthly Weather Review), or the acronym used in the 
		associated paper (e.g., AIES) is used.
	4) Year of publication
		The year the article was posted online/in print.
	5) Is there an ASDC?
		If the article contains an availability statement in any form, "yes" is 
		recorded. Otherwise, "no" is recorded.
	6) Justification for non-open data?
		If an availability statement contains some justification for why data is not 
		openly available, the justification is summarized and recorded as one of the 
		following options: 1) Dataset too large, 2) Licensing/Proprietary, 3) Can be 
		obtained from other entities, 4) Sensitive information, 5) Available at later 
		date. If the statement indicates any data is not openly available and no 
		justification is provided, or if no statement is provided is provided "None" 
		is recorded. If the statement indicates openly available data or no data 
		produced, "N/A" is recorded.
	7) All data available
		If there is an availability statement and data is produced, "y" is recorded 
		if means to access data associated with the article are given and there is no 
		indication that any data is not openly available; "n" is recorded if no means 
		to access data are given or there is some indication that some or all data is 
		not openly available. If there is no availability statement or no data is 
		produced, the record is left blank.
	8) At least some data available
		If there is an availability statement and data is produced, "y" is recorded 
		if any means to access data associated with the article are given; "n" is 
		recorded if no means to access data are given. If there is no availability 
		statement or no data is produced, the record is left blank.
	9) All code available
		If there is an availability statement and data is produced, "y" is recorded 
		if means to access code associated with the article are given and there is no 
		indication that any code is not openly available; "n" is recorded if no means 
		to access code are given or there is some indication that some or all code is 
		not openly available. If there is no availability statement or no data is 
		produced, the record is left blank.
	10) At least some code available
		If there is an availability statement and data is produced, "y" is recorded 
		if any means to access code associated with the article are given; "n" is 
		recorded if no means to access code are given. If there is no  availability 
		statement or no data is produced, the record is left blank.
	11) All data available upon request
		If there is an availability statement indicating data is produced and no data 
		is openly available, "y" is recorded if any data is available upon request to 
		the authors of the examined journal article (not a request to any other 
		entity); "n" is recorded if no data is available upon request to the authors 
		of the examined journal article. If there is no availability statement, any 
		data is openly available, or no data is produced, the record is left blank.
	12) At least some data available upon request
		If there is an availability statement indicating data is produced and not all 
		data is openly available, "y" is recorded if all data is available upon 
		request to the authors of the examined journal article (not a request to any 
		other entity); "n" is recorded if not all data is available upon request to 
		the authors of the examined journal article. If there is no availability 
		statement, all data is openly available, or no data is produced, the record
		is left blank.
	13) no data produced
		If there is an availability statement that indicates that no data was
		produced for the examined journal article, "y" is recorded. Otherwise, the
		record is left blank.
	14) links work
		If the availability statement contains one or more links to a data or code 
		repository, "y" is recorded if all links work; "n" is recorded if one or more 
		links do not work. If there is no availability statement or the statement 
		does not contain any links to a data or code repository, the record is left 
		blank. 