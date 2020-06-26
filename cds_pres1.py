import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'reanalysis',
        'format':'netcdf',
        'area'  : '10/40/40/70',
        'variable':[
            'geopotential','relative_humidity','specific_humidity',
            'temperature','u_component_of_wind','v_component_of_wind'
        ],
        'pressure_level':[
            '100','200','500',
            '850','900'
        ],
        'year':'2017',
        'month':'12',
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'time':[
            '06:00',
            
        ]
    },
    'download06.nc')
