# apollo

Currently working on the dataset construction: Order
1.  DataAssembly/company_names/extract() 
    [generate initial S&P company list]
2.  DataAssembly/build_pricing_db.py/create_pricing_db 
    [extract pricing for companies on list with sufficient data]
3.  DataAssembly/company_names/see_which_have_existed_for_long_enough() 
    [remove companies from metadata list that don't have enough data (syncing price db and metadata csv)]
4.  DataAssembly/collect_earnings_transcript.py/add_missing_tickers()
    [retrieves URLs for 10-Q's + filed and reporting for dates for companies in DB]
5.  DataAssembly/collect_earnings_transcript_analysis.py/main()
    [retries tickers who retrieved insuffcient/too many 10-Q's, removes tickers that reproduce guff from metadata and pricing data]
6.  EPS Data will be pre-prepared, its to far to slow and the parsing process isn't robust enough to fully automate this
    This will also be the case for revenus data once this pipeline is prepared.
7.  DataAssembly/NewsArticles/restore_row_structure.py
    [This downloads the raw new data if not present and converts it from its very very messy and malformed structure into rows]
    [This also removes rows relating to tickers that aren't being analysed according the the 'Ticker' col in eps_data.csv and 'Symbol' in news.csv]

