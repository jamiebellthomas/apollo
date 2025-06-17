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

PAUSE: At this point we have a syncronised list of tickers between Data/momentum_data.db/daily_prices, Data/filing_dates_and_urls.csv, and Data/filtered_sp500_metadata.csv. Now we need to syncronise the two CSVs into the database in seperate tables. 
