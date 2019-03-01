from tkinter import *
import datetime
from dateutil import parser
import pickle
import pandas as pd
import numpy as np


def cleaninputs(rawinputs): 
    '''
    Rawinputs is a dictionary from the GUI mapping raw user input fields to the entered values. 
    Returns a 1-row dataframe containing cleaned inputs and additional fields. 
    '''    
    timeblocks= ['00:00-05:59', '06:00-09:59', '10:00-14:59', '15:00-18:59', '19:00-23:59']
    clean= pd.DataFrame() #Create empty dataframe
    
    clean.loc[0, 'R']= rawinputs['Randomness %']
    clean['S']= rawinputs['Swipe %']
    clean['date']= rawinputs['Date'] #this is already a datetime.datetime
    clean['timeblock']= rawinputs['Timeblock'] #timeblock as a string
    clean['blocknum']= timeblocks.index(rawinputs['Timeblock'])+1 #timeblock as an int
    with open('training_range.pkl', 'rb') as daterange: #load the dates by previous training session
        training_range= pickle.load(daterange)
    clean['year#']= int((clean['date'][0] - training_range[0]).days/365.2422) #years since data start
    clean['TMAX']= rawinputs['High temp'] #already an int
    clean['wkday']= rawinputs['Date'].weekday() #as an int
    wkblocknum= clean['blocknum'][0] + 5*clean['wkday'][0] #used to determine iswknd
    clean['iswknd']= int(23 < wkblocknum <= 33) #whether that timeblock counts as a weekend
    
    binary= pd.DataFrame({'wkday': list(str(i) for i in range(7))+[str(clean['wkday'][0])]})
    wkdaybinary= pd.DataFrame(pd.get_dummies(binary, prefix='wkday').iloc[-1]).T.reset_index(drop=True)
    clean= pd.concat([clean, wkdaybinary], axis=1) #adds binary format of wkdays

    return clean


def swipemodel(cleaninputs):
    '''
    Given clean input data, returns predictions for swipes from the means and random forest stored in directory. 
    '''
    from sklearn.ensemble import RandomForestRegressor

    allzbwmeans= np.load('allzbwmeans.npy')
    
    with open('swipeforest.pkl','rb') as handle:
        regr= pickle.load(handle)
    
    for i in range(len(pd.read_csv('zone_to_desc_mapping.csv'))):
        xtest= {'year#':cleaninputs['year#'],'TMAX':cleaninputs['TMAX']}
        xtest['avg']= allzbwmeans[i, cleaninputs['blocknum']-1, cleaninputs['wkday']]
        xtest= pd.DataFrame(data=xtest)
        cleaninputs['swipe_z'+str(i+1)]= regr.predict(xtest)
    
    return cleaninputs


def crimemodel(cleaninputs):
    '''
    Given clean input data, returns predictions for crimes from the means and linear regression stored in directory. 
    '''
    from sklearn.linear_model import LinearRegression
    
    cleaninputs= swipemodel(cleaninputs)
    allzbmeans= np.load('allzbmeans.npy')
    
    with open('crimelinreg.pkl','rb') as handle:
        regr= pickle.load(handle)
    
    for i in range(len(pd.read_csv('zone_to_desc_mapping.csv'))):
        xtest= {'year#':cleaninputs['year#'],'TMAX':cleaninputs['TMAX']}
        xtest['iswknd']= cleaninputs['iswknd']
        xtest['swipes']= cleaninputs['swipe_z'+str(i+1)]
        xtest['avg']= allzbmeans[i, cleaninputs['blocknum']-1]
        xtest= pd.DataFrame(data=xtest)
        
        cleaninputs['crime_z'+str(i+1)]= regr.predict(xtest)
    
    return cleaninputs


def prioritized(allinputs): 
    '''
    vals: dictionary mapping the names of the model inputs to their values.  

    Returns a DataFrame. 'zone' column is a list of all zones in order of priority based on the 
    crime and swipe models and the vales of s and r. 'val' column contains the priority metric 
    associated with each zone. This value aggregates expcrimes, expswipes, and s, but NOT r. So, 
    depending on r's value, the list's order should reflect 'val' (unless r==100), but will not 
    necessarily be in perfect descending sorted order by 'val'. 
    '''
    s, r= allinputs['S'][0]/100, allinputs['R'][0]/100 #we want s and r in 0..1 rather than 0..100
    
    r= r**10 #The power here changes how quickly increasing r increases the randomness. 
    #Higher power = less randmoness. This should attempt to match a user's expectations. 
    
    #expcrimes: dataframe: columns= [zone, expected crime metric]. Get this from crimemodel(cleaninputs). 
    #expswipes: dataframe: columns= [zone, expected number of swipes]. Get this from swipemodel(cleaninputs).
    swipecols= ['swipe_z'+str(i+1) for i in range(len(pd.read_csv('zone_to_desc_mapping.csv')))]
    crimecols= ['crime_z'+str(i+1) for i in range(len(pd.read_csv('zone_to_desc_mapping.csv')))]
    zonenames= sorted([str(i+1) for i in range(len(pd.read_csv('zone_to_desc_mapping.csv')))])
    expswipes= allinputs[swipecols].sort_index(axis=1)
    expswipes.columns= zonenames
    expswipes= expswipes.T.sort_index()
    expcrimes= allinputs[crimecols].sort_index(axis=1)
    expcrimes.columns= zonenames
    expcrimes= expcrimes.T.sort_index()
    
    #both: dataframe containing expected swipes and crimes for each station. 
    both= pd.concat([expswipes, expcrimes], axis=1).reset_index()
    both.columns=['zone','expswipes','expcrimes']
    
    #For each zone, convert expcrimes and expswipes to percentage of total crimes and swipes. 
    both[['expswipes','expcrimes']] /= np.sum(both[['expswipes','expcrimes']], axis=0)

    def swipeweight(row): #Formula for combining swipes and crimes, weighted by the value of s. 
        return row['expswipes']**s * row['expcrimes']**(1-s)

    both['val']= both.apply(swipeweight, axis=1) #priority metric
    both= both[['zone','val']]
    both['val'] /= np.sum(both['val']) #scale priority metrics to percentage of total

    if r==0: #If no randomness, just sort by the priority metric and return. 
        return both.sort_values(by='val', ascending=False).reset_index(drop=True)

    else: #a randomness element is involved
        bests= pd.DataFrame(columns=both.columns)
        ogboth= both.copy(deep=True)
        r= 1 + np.log10(r) #transform r so that it can be used as a power
        both['val'] **= (1-r) #stretch or squeeze priority metrics based on r
        while len(both > 0): #do this numzones times:
            both['val'] /= np.sum(both['val']) #scale priority metrics to sum of total
            #semi-randomly choose a zone using priority metrics as probability distribution
            best= (np.random.choice(both.index.values, p=both['val']))
            bests= bests.append(ogboth.loc[best], ignore_index=True)
            both= both.drop(labels=[best]) #use this zone as the next priority
        bests['val'] /= np.sum(bests['val'])
        return bests #return semi-randomized list of zones by priority


def gethightemp(date): 
    '''
    Gets a datetime, returns high temp from that date as a string. 
    If date does not fall within 10-day forecast, returns None. 
    '''
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as et
    import datetime
    from dateutil import parser
    
    if date is None:
        return None
    
    URL= 'https://query.yahooapis.com/v1/public/yql?q=select+%2A+from+weather.forecast+where+woeid%3D28288749&format=xml'
    xmlText= urllib.request.urlopen(URL).read()
    root= et.fromstring(xmlText)
    alldays= root.find('results').find('channel').find('item').findall('{http://xml.weather.yahoo.com/ns/rss/1.0}forecast')
    hightemp= None
    for day in alldays: #Find the forecast that matches 'date' and use that high temp.
        newdate= parser.parse(day.get('date'))
        if date.date()==newdate.date():
            hightemp= day.get('high')
            break
    
    return hightemp


def rungui():    
    zone_desc= pd.read_csv('zone_to_desc_mapping.csv') #description of each zone
    numzones= len(zone_desc) #number of zones
    
    allfield= ['Randomness %', 'Swipe %','Date','Timeblock','High temp', 'Show top:'] #all entry fields
    alllabel= {} #Label objects next to entry boxes
    allentry= {} #entry boxes
    allerror= {} #error messages
    rawinputs={} #values entered
    ready= False #True when all inputs are valid and ready for processing

    master= Tk() #main window
    master.title('Output Tool')

    with open('training_range.pkl', 'rb') as daterange: #load the dates by previous training session
        training_range= pickle.load(daterange)
    trainedtext= 'Last trained on '+str(training_range[0].date())+'..'+str(training_range[1].date())+' data.'
    lasttrained= Label(master, text=trainedtext, fg='gray')
    lasttrained.grid(row=0, column=1, columnspan=3) #display date last trained on
    
    #Checkbox, whether to display zone descriptions in results
    verbose= IntVar(master)
    verbosebox= Checkbutton(master, text='Show zone details', variable=verbose)
    verbosebox.grid(row=1, column=2, sticky=W+E)

    #field labels, entry boxes, error messages
    for i in range(len(allfield)): 
        field= allfield[i]
        alllabel[field]= Label(master, text=field)
        alllabel[field].grid(row=i+3, column=1, sticky=E)
        allentry[field]= Entry(master)
        allentry[field].grid(row=i+3, column=2)
        allerror[field]= Label(master, text='', fg='red')
        allerror[field].grid(row=i+3, column=3, sticky=W)
            
    #timeblock dropdown menu
    timeblocks= ['Full day', '00:00-05:59', '06:00-09:59', '10:00-14:59', '15:00-18:59', '19:00-23:59']
    blockpos= 3
    allentry['Timeblock'].destroy()
    blockchoice= StringVar(master)
    allentry['Timeblock']= OptionMenu(master, blockchoice, *timeblocks)
    allentry['Timeblock'].grid(row=blockpos+3, column=2, sticky=W+E)
        
    def clearall():
        '''
        clear all entry boxes
        '''
        for field in allfield:
            if type(allentry[field])==Entry:
                allentry[field].delete(0,END)
            if type(allentry[field])==OptionMenu:
                blockchoice.set('')
        verbose.set(False)
                    
    def restoredefs(): 
        '''
        set all entry boxes to their defaults
        '''
        clearall()
        allentry['Randomness %'].insert(0,'0.0')
        allentry['Swipe %'].insert(0,'0.0')
        allentry['Show top:'].insert(0,'10')
        now= datetime.datetime.now().date()
        allentry['Date'].insert(0,str(now+datetime.timedelta(days=1))) #use tomorrow
        blockchoice.set(timeblocks[0])
        verbose.set(True)
    
    def setblock(field): 
        '''
        set the timeblock based on user input. Return whether successful. 
        '''
        if blockchoice.get() in timeblocks:
            rawinputs[field]= blockchoice.get()
            allerror[field]['text']= '' #no error
            return True
        else:
            allerror[field]['text']= 'Select a block. '
            return False

    def setslider(field): 
        '''
        set a 0..100 field based on user input. Return whether successful. 
        '''
        entry= allentry[field].get()
        try: 
            entry= float(entry)
            assert 0<= entry <= 100
            rawinputs[field]= entry
            allerror[field]['text']= '' #no error
            return True
        except: 
            allerror[field]['text']= 'Must be number in 0..100'
            return False
    
    def showtop(field): 
        '''
        set x to display the top x priority stations
        '''
        entry= allentry[field].get()
        try: 
            entry= int(entry)
            assert 1<= entry <= numzones
            rawinputs[field]= entry
            allerror[field]['text']= '' #no error
        except: 
            allerror[field]['text']= 'Must be an integer in 1..'+str(numzones)
    
    def settemp(field): 
        '''
        check that temperature entered is a number
        '''
        entry= allentry[field].get()
        try: 
            entry= float(entry)
            rawinputs[field]= entry
            allerror[field]['text']= ''
        except: 
            allerror[field]['text']= 'Must be a number'
            
    def setdate(field): 
        '''
        set date from user input
        '''
        entry= allentry[field].get()
        try: 
            rawinputs[field]= parser.parse(entry) #susceptible to misinterpretation since parser.parse is loose
            allerror[field]['text']= ''
            return rawinputs[field]
        except: 
            allerror[field]['text']= 'Must be a date. '

    def loadweather(field='High temp'): 
        '''
        load the weather from the forecast
        '''
        date= setdate('Date')
        allentry[field].delete(0,END)
        temp= gethightemp(date)
        if temp is not None: #if temperature is not in 10 day forecast
            allentry[field].insert(0,temp)
        else: 
            allerror[field]['text']= 'Date not in 10-day forecast.'
    
    def setall(): 
        '''
        when Get Results is pressed, attempt to set all fields' values
        '''
        setslider('Randomness %')
        setslider('Swipe %')
        settemp('High temp')
        setblock('Timeblock')
        setdate('Date')
        showtop('Show top:')
        ready= True
        for msg in allerror.values(): #if there's an error, don't proceed
            if msg['text']!='':
                ready= False
        if ready: #if no errors, proceed
            resultwindow()
                
    def resultwindow(): 
        '''
        creates the results window, loads results in
        '''
        results= Toplevel()
        results.title('Results')

        if blockchoice.get()=='Full day': #for full day, show 5 timeblocks
            for i in range(5):
                rawinputs['Timeblock']= timeblocks[i+1]
                prediction= prioritized(crimemodel(cleaninputs(rawinputs)))
                showresults(prediction, results, i*4)
        else: #for 1 day, just show 1 timeblock
            prediction= prioritized(crimemodel(cleaninputs(rawinputs)))
            showresults(prediction, results, 0)
            
    def showresults(prediction, results, col): #sets up the actual view of results
        '''
        Prediction is the dataframe of model results
        Results is the result window
        Col is the column at which to start displaying results
        '''
        allresults= {}
        allranks= {}
        allvals= {}
        
        if col != 0: #if multiple results, then include separating lines
            import tkinter.ttk
            line= tkinter.ttk.Separator(results, orient=VERTICAL)
            line.grid(column=col, row=0, rowspan=rawinputs['Show top:']+4, sticky='ns')
        
        #Show the entry values for the user to see
        import calendar
        date= rawinputs['Date']
        block= rawinputs['Timeblock']
        if col == 0: #if first column, show everything
            datetext= calendar.day_abbr[date.weekday()]+' '+str(date.date())+' '+block
        else: datetext= block #if not first column, only show the timeblock
        dtm= Label(results, text=datetext, fg='gray')
        dtm.grid(row=0, column=col+0, columnspan=3)
        
        if col == 0: #if it's the first column, show EVERYTHING
            rstext='R% = '+str(rawinputs['Randomness %'])+', S% = '+str(rawinputs['Swipe %'])
            rs= Label(results, text=rstext, fg='gray')
            rs.grid(row=1, column=col+0, columnspan=3)

            temptxt='High temperature = '+str(rawinputs['High temp'])
            temp= Label(results, text=temptxt, fg='gray')
            temp.grid(row=2, column=col+0, columnspan=3)

        if col == 0: #if first column, then have a ranking column
            rankhead= Label(results, text='Priority', font='-size 9 -underline True')
            rankhead.grid(row=3, column=col+0)
            
        #Label the columns
        zonehead= Label(results, text='Zone', font='-size 9 -underline True')
        zonehead.grid(row=3, column=col+1)
        zonehead= Label(results, text='Index', font='-underline True -size 9')
        zonehead.grid(row=3, column=col+2)
                            
        for i in range(rawinputs['Show top:']): #for each row we're displaying:
            if col == 0: #load the ranks if first col
                allranks[i]= Label(results, text= str(i+1)+'. ')
                allranks[i].grid(row=i+4, column=col+0, sticky=E)
            content= 'Zone ' + str(prediction['zone'][i])
            if verbose.get(): #if verbose, include descriptions
                content += ': '+zone_desc.loc[zone_desc['Zone']==int(prediction['zone'][i]),'Desc'].values[0]
            allresults[i]= Label(results, text=content)
            allresults[i].grid(row=i+4, column=col+1, sticky=W)
            allvals[i]= Label(results, text=str(np.round(prediction['val'][i]*100,1)))
            allvals[i].grid(row=i+4, column=col+2)
    
    #create buttons and their corresponding commands
    getbutton= Button(master, text='Get Results', command=setall)
    getbutton.grid(row=len(allfield)+3,column=1,columnspan=3,sticky=W+E)
    restorebutton= Button(master, text='Use Defaults', command=restoredefs)
    restorebutton.grid(row=len(allfield)+4,column=1, sticky=W+E)
    clearbutton= Button(master, text='Clear All', command=clearall)
    clearbutton.grid(row=len(allfield)+4,column=3, sticky=W+E)
    weatherbutton= Button(master, text='Use weather forecast', command=loadweather)
    weatherbutton.grid(row=len(allfield)+4,column=2, sticky=W+E)
    
    mainloop()

rungui()