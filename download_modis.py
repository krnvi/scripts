import os
import requests
import datetime
print """
########################################################################################
###  CHANGE ME! CHANGE ME! CHANGE ME! CHANGE ME! CHANGE ME! CHANGE ME! CHANGE ME!    ###
########################################################################################
"""
username = "Olliewolly"
password = "Skoa1319!!"

def url_downloader (username, password, tile, destination_folder="data/",
                    url="https://n5eil01u.ecs.nsidc.org/DP5/MOST/MOD10A1.006/2016.01.01/"):
    """Downloads data from a NASA URL, provided that a username/password pair exist.
    Parameters
    ----------
    username: str
        The NASA EarthData username
    password: str
        The NASA EarthData password
    destination_folder: str
        The destination folder
    url: str
        The required URL
    
    Returns
    --------
    A string with the location of the downloaded file.
    """
    with requests.Session() as session:
            session.auth = (username, password)
            r1 = session.request('get', url)
            r = session.get(r1.url, auth=(username, password))
            if r.ok:
                for line in r.text.split("\n"):
                    if line.find(tile) >= 0 and line.find(".hdf") >= 0 and line.find(".xml") < 0:
                        fname = line.split("href")[1][1:].split('"')[1]
            url_granule = url + fname
            r1 = session.request('get', url_granule)
            r = session.get(r1.url, auth=(username, password))
            output_fname = os.path.join(destination_folder, fname)
            if r.ok:
                with open(output_fname, 'w') as fp:
                    fp.write(r.content)
            print "Saved file {}".format(output_fname)
            return output_fname
# Test downloading a file
the_file = url_downloader (username, password, "h09v05")
# check file is there
import gdal
g = gdal.Open(the_file)
print g