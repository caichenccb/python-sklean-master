# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:39:09 2019

@author: 92156
"""

from statsmodels.imputation import mice
imp = mice.MICEData(data)
>>> fml = 'y ~ x1 + x2 + x3 + x4'
>>> mice = mice.MICE(fml, sm.OLS, imp)
>>> results = mice.fit(10, 10)
>>> print(results.summary())