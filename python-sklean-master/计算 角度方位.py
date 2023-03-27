# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:34:45 2019

@author: 92156
"""
#城市就三个  北京上海 广州

import ephem 
u = ephem.Uranus() 
# 天王星
u.compute('2010/1/16') 
print (u.ra, u.dec, u.mag) 
# 赤经、赤纬、亮度
print (ephem.constellation(u)) 
# 所在星座
# print (u.rise_time)



import ephem
j = ephem.Jupiter('2010/1/16') 
# 木星
n = ephem.Neptune('2010/1/16') 
# 海王星
print ("Jupiter") 
print ("RA:", j.ra, ", \nDEC", j.dec, ", \nMAG:", j.mag)
print ("Neptune")
print ("RA:", n.ra, ", \nDEC:", n.dec, ", \nMAG:", n.mag)
print ("Separation between Jupiter and Neptune:", ephem.separation(j, n)) 
# 木星和海王星的角距

#计算月球在近日点和远日点速度的差异
import ephem

def hpos(body): return body.hlong, body.hlat

ma0 = ephem.Moon('2016/02/21')    
# 月球在远日点
ma1 = ephem.Moon('2016/02/21')
print (ephem.separation(hpos(ma0), hpos(ma1)))
mp0 = ephem.Moon('2016/02/21')    
# 月球在远日点
mp1 = ephem.Moon('2016/02/21')
print (ephem.separation(hpos(mp0), hpos(mp1)))

import ephem
d = ephem.Date('1984/12/21 15:00')
ephem.localtime(d) 
# 地方时print (ephem.localtime(d).ctime())

import ephem
gatech = ephem.Observer() 
gatech.long, gatech.lat = ("39.7867960000","116.4691270000")
gatech.date = ('2016/2/22 16:20:56')
sun, moon = ephem.Sun(), ephem.Moon() 
sun.compute(gatech)
for i in range(8):
    old_az, old_alt = sun.az, sun.alt
    gatech.date += ephem.minute * 5.
    sun.compute(gatech)
    sep = ephem.separation((old_az, old_alt), (sun.az, sun.alt))
    print("%s %s %s" % (gatech.date, sun.alt, sep))
    
    
    
    
import ephem

#defining an observer
obs = ephem.Observer()

#defining position
long = '39.7867960000'
lat = '116.4691270000'

obs.long = ephem.degrees(long)
obs.lat = ephem.degrees(lat)

print ("long = ", obs.long, "lat = ", obs.lat)

#defining date
date = '2016/02/22'

obs.date = ephem.Date(date)

#defining an astronomic object; Sun in this case
sun = ephem.Sun(obs)

r1 = obs.next_rising(sun)
s1 = obs.next_setting(sun)

print ("rising sun : ", r1)
print ("setting sun : ", s1)

r1_lt = ephem.Date(r1 - 6 * ephem.hour) #local time 

(y, mn, d, h, min, s) = r1_lt.tuple()

print ("rising sun: (local time): {:.2f}".format( h + min/60. + s/3600. ))

s1_lt = ephem.Date(s1 - 6 * ephem.hour) #local time

(y, mn, d, h, min, s) = s1_lt.tuple()

print ("setting sun (local time): {:.2f}".format( h + min/60. + s/3600. ))
#计算日落日升
import ephem
sun = ephem.Sun()
greenwich = ephem.Observer()
#city = ephem.city('Beijing')
greenwich.long, greenwich.lat = ("39.7867960000","116.4691270000")   #纬度  #经度
greenwich.date = '2019/08/29'
r1 = greenwich.next_rising(sun)
r3 = greenwich.next_setting(sun)
greenwich.horizon = "-8"
greenwich.date = '2016/02/22'
r2 = greenwich.next_rising(sun)
r4 = greenwich.next_setting(sun)
print('Visual sunrise: %s' % r1)
print('Visual sunset: %s' % r3)

print('Naval Observatory sunrise: %s' % r2)
print('Naval Observatory sunset: %s' % r4)


import ephem 

#Make an observer 
fred  = ephem.Observer() 

#PyEphem takes and returns only UTC times. 15:00 is noon in Fredericton 
fred.date = "2016-02-22 00:00:00" 

#Location of Fredericton, Canada 
fred.lon = str(116.40 ) #Note that lon should be in string format 
fred.lat = str(39.40)  #Note that lat should be in string format 

#Elevation of Fredericton, Canada, in metres 
fred.elev = 20 

#To get U.S. Naval Astronomical Almanac values, use these settings 
fred.pressure= 0 
fred.horizon = '-0:34' 

sunrise=fred.previous_rising(ephem.Sun()) #Sunrise 
noon =fred.next_transit (ephem.Sun(), start=sunrise) #Solar noon 
sunset =fred.next_setting (ephem.Sun()) #Sunset 

#We relocate the horizon to get twilight times 
fred.horizon = '-6' #-6=civil twilight, -12=nautical, -18=astronomical 
beg_twilight=fred.previous_rising(ephem.Sun(), use_center=True) #Begin civil twilight 
end_twilight=fred.next_setting (ephem.Sun(), use_center=True) #End civil twilight 
print(sunrise)
print(noon)
print(sunset)
print(beg_twilight)
print(end_twilight)


#最好的
import ephem
import datetime

Boston=ephem.Observer()
Boston.lat='39.05'
Boston.lon='116'
Boston.date ="2019/8/29"
Boston.elevation = 3 # meters
Boston.pressure = 1010 # millibar
Boston.temp = 25 # deg. Celcius
Boston.horizon = 0

sun = ephem.Sun()
moon=ephem.Moon()
print("Next sunrise in Boston will be: ",ephem.localtime(Boston.next_rising(sun)))
print("Next sunrise in Boston will be: ",ephem.localtime(Boston.previous_rising(sun)))

print("Next sunset in Boston will be: ",ephem.localtime(Boston.next_setting(sun)))
print("Next moonrise in Boston will be: ",ephem.localtime(Boston.next_rising(moon)))
print("Next moonset in Boston will be: ",ephem.localtime(Boston.next_setting(moon)))

#Next sunrise in Boston will be:  2016-02-23 06:58:10.800156
#Next sunset in Boston will be:  2016-02-22 18:00:11.528084


#计算每天 月亮升起 落下
import ephem
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroplan import moon_illumination

from icalendar import Calendar, Event
from datetime import date, time, timedelta, datetime

my_lat = '24.88554'
my_lon = '102.82147'
my_elev = 122
date_start = '2017/01/01 00:00'

obs = ephem.Observer()
moon = ephem.Moon()

obs.lat = my_lat
obs.lon = my_lon
obs.elevation = my_elev
obs.date = date_start

loc = EarthLocation(lat=my_lat, lon=my_lon)

moonrise_all = []
moonset_all = []
illum_all = []

for x in range(0,365):
# for x in range(0,1):
    print("Calculation for {0}:".format(obs.date))

    moon.compute(obs)
    if (moon.alt > 0):
        print("    Moon is currently up")
        moon_up = True
        moonrise = obs.previous_rising(moon)
        moonset = obs.next_setting(moon)
    else:
        print("    Moon is currently down")
        moon_up = False
        moonrise = obs.next_rising(moon)
        moonset = obs.next_setting(moon)


    illum = moon_illumination(Time(moonrise.datetime()))*100

    moonrise_all.append(ephem.localtime(moonrise))
    moonset_all.append(ephem.localtime(moonset))
    illum_all.append(illum)

    print("    Moonrise: {0}".format(ephem.localtime(moonrise)))
    print("    Moonset:  {0}".format(ephem.localtime(moonset)))
    print("    Illum:    {0:.0f}%".format(illum))

    obs.date = obs.date + 1

# ical stuff starts here
cal = Calendar()
cal.add('prodid', '-//python icalendar//python.org//')
cal.add('version', '2.0')

for r, s, i in zip(moonrise_all, moonset_all, illum_all):
    # moonrise event
    e1 = Event()
    moonrise_simpletime = time.strftime(r.time(), "%H:%M")
    e1.add('uid', "{0}@curbo.org".format(r.isoformat()))
    e1.add('summary', "Moonrise at {0}, illum {1:.0f}%".format(moonrise_simpletime, i))
    e1.add('dtstart', r)
    e1.add('dtend', r + timedelta(minutes=15))
    e1.add('dtstamp', datetime.now())
    cal.add_component(e1)

    # moonset event
    e2 = Event()
    moonset_simpletime = time.strftime(s.time(), "%H:%M")
    e2.add('uid', "{0}@curbo.org".format(s.isoformat()))
    e2.add('summary', "Moonset at {0}, illum {1:.0f}%".format(moonset_simpletime, i))
    e2.add('dtstart', s)
    e2.add('dtend', s + timedelta(minutes=15))
    e2.add('dtstamp', datetime.now())
    cal.add_component(e2)

# write out the ics file
f = open('moon.ics', 'wb')
f.write(cal.to_ical())
f.close()

#计算每天 日升起 落下
import ephem
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroplan import moon_illumination
from icalendar import Calendar, Event
from datetime import date, time, timedelta, datetime

my_lat = '39.916527'
my_lon = '116.397128'
my_elev = 122
date_start = '2016/01/01 00:00'

obs = ephem.Observer()
sun = ephem.Sun()

obs.lat = my_lat
obs.lon = my_lon
obs.elevation = my_elev
obs.date = date_start

loc = EarthLocation(lat=my_lat, lon=my_lon)
a_all=[]
b_all=[]
sunrise_all = []
sunset_all = []
illum_all = []

for x in range(0,365):
# for x in range(0,1):
    print("Calculation for {0}:".format(obs.date))

    sun.compute(obs)
    if (sun.alt > 0):
        print("    Sun is currently up")
        sun_up = True
        sunrise = obs.previous_rising(sun)
        sunset = obs.next_setting(sun)
        
    else:
        print("    Sun is currently down")
        sun_up = False
        sunrise = obs.next_rising(sun)
        sunset = obs.next_setting(sun)


    

    sunrise_all.append(ephem.localtime(sunrise))
    sunset_all.append(ephem.localtime(sunset))
    
    a_all.append(ephem.localtime(sunrise))
    b_all.append(ephem.localtime(sunset))
    print("    sunrise: {0}".format(ephem.localtime(sunrise)))
    print("    sunset:  {0}".format(ephem.localtime(sunset)))
#    print("    Illum:    {0:.0f}%".format(illum))

    obs.date = obs.date + 1

# ical stuff starts here
cal = Calendar()
cal.add('prodid', '-//python icalendar//python.org//')
cal.add('version', '2.0')

for r, s, i in zip(sunrise_all, sunset_all, illum_all):
    # moonrise event
    e1 = Event()
    sunrise_simpletime = time.strftime(r.time(), "%H:%M")
    e1.add('uid', "{0}@curbo.org".format(r.isoformat()))
    e1.add('summary', "Moonrise at {0}, illum {1:.0f}%".format(sunrise_simpletime, i))
    e1.add('dtstart', r)
    e1.add('dtend', r + timedelta(minutes=15))
    e1.add('dtstamp', datetime.now())
    cal.add_component(e1)

    # moonset event
    e2 = Event()
    sunset_simpletime = time.strftime(s.time(), "%H:%M")
    e2.add('uid', "{0}@curbo.org".format(s.isoformat()))
    e2.add('summary', "Moonset at {0}, illum {1:.0f}%".format(sunset_simpletime, i))
    e2.add('dtstart', s)
    e2.add('dtend', s + timedelta(minutes=15))
    e2.add('dtstamp', datetime.now())
    cal.add_component(e2)

# write out the ics file
f = open('moon.ics', 'wb')
f.write(cal.to_ical())
f.close()
import pandas as pd
test=pd.DataFrame(data=b_all)#数据有三列，列名分别为one,two,three
test1=pd.DataFrame(data=a_all)#数据有三列，列名分别为one,two,three
test.to_excel("riluo.xls")
test1.to_excel("richu.xls")

#仰角+月出时间
#计算每天 月亮升起 落下
import ephem
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroplan import moon_illumination
from ephem  import *
from icalendar import Calendar, Event
from datetime import date, time, timedelta, datetime

my_lat = '43.36378'
my_lon = '88.31104'
my_elev = 122
date_start = "2016/1/1"
obs = ephem.Observer()
moon = ephem.Moon()

obs.lat = my_lat
obs.lon = my_lon
obs.elevation = my_elev
obs.date = date_start

loc = EarthLocation(lat=my_lat, lon=my_lon)
yangjiao=[]
a_all=[]
b_all=[]
d_all=[]
c_all=[]
moonrise_all = []
moonset_all = []
illum_all = []
moontransit_all=[]

for x in range(0,365):
# for x in range(0,1):
#    print("Calculation for {0}:".format(obs.date))
   
    moon.compute(obs)
    if (moon.alt > 0):
#        print("    Moon is currently up")
        moon_up = True
        moonrise = obs.previous_rising(moon)
        moonset = obs.next_setting(moon)
        moontransit=obs.next_transit(moon)
    else:
#        print("    Moon is currently down")
        moon_up = False
        moonrise = obs.next_rising(moon)
        moonset = obs.next_setting(moon)
        moontransit=obs.next_transit(moon)

    illum = moon_illumination(Time(moonrise.datetime()))*100

#    moonrise_all.append(ephem.localtime(moonrise))
#    moonset_all.append(ephem.localtime(moonset))
    moontransit_all.append(ephem.localtime(moontransit))
    illum_all.append(illum)
#    a_all.append(ephem.localtime(moonrise))
#    b_all.append(ephem.localtime(moonset))
    c_all.append(illum)
#    d_all.append(ephem.localtime(moontransit))
#    print("    Moonrise: {0}".format(ephem.localtime(moonrise)))
#    print("    Moonset:  {0}".format(ephem.localtime(moonset)))
#    print("    Moontransit:  {0}".format(ephem.localtime(moontransit)))
#    print("    Illum:    {0:.0f}%".format(illum))
    v = Moon(obs)
    v=float(v.alt)
#    print("    alt: {0}".format(v))
    yangjiao.append(v)
    obs.date = obs.date + 1

# ical stuff starts here
cal = Calendar()
cal.add('prodid', '-//python icalendar//python.org//')
cal.add('version', '2.0')

for r, s, i in zip(moonrise_all, moonset_all, illum_all):
    # moonrise event
    e1 = Event()
    moonrise_simpletime = time.strftime(r.time(), "%H:%M")
    e1.add('uid', "{0}@curbo.org".format(r.isoformat()))
    e1.add('summary', "Moonrise at {0}, illum {1:.0f}%".format(moonrise_simpletime, i))
    e1.add('dtstart', r)
    e1.add('dtend', r + timedelta(minutes=15))
    e1.add('dtstamp', datetime.now())
    cal.add_component(e1)

    # moonset event
    e2 = Event()
    moonset_simpletime = time.strftime(s.time(), "%H:%M")
    e2.add('uid', "{0}@curbo.org".format(s.isoformat()))
    e2.add('summary', "Moonset at {0}, illum {1:.0f}%".format(moonset_simpletime, i))
    e2.add('dtstart', s)
    e2.add('dtend', s + timedelta(minutes=15))
    e2.add('dtstamp', datetime.now())
    cal.add_component(e2)

# write out the ics file
f = open('moon.ics', 'wb')
f.write(cal.to_ical())
f.close()

import pandas as pd
#test=pd.DataFrame(data=b_all)#数据有三列，列名分别为one,two,three
#test1=pd.DataFrame(data=a_all)#数据有三列，列名分别为one,two,three
test2=pd.DataFrame(data=c_all)#数据有三列，列名分别为one,two,three
#test3=pd.DataFrame(data=yangjiao)#数据有三列，列名分别为one,two,three
#test4=pd.DataFrame(data=d_all)#数据有三列，列名分别为one,two,three
#test.to_excel("yueluo.xls")
#test1.to_excel("yuechu.xls")
test2.to_excel("liangdu.xls")
#test3.to_excel("yangjiao.xls")
#test4.to_excel("yuezhong.xls")



