#!/usr/bin/env python
# coding: utf-8

# **EDA - Fraud Detection**
# 
# In this problem, we are trying to predict the Ad clicks that lead to App download. These clicks are considered Non-Fraudulent. So I may use the terms App Downloaded and Non-Fraudulent Clicks interchangeably
# 
# The training dataset is quite huge - 185 million. 
# 

# 

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15355/20428/Screen%20Shot%202018-03-07%20at%203.33.06%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1520999951&Signature=OzlkiJD0c%2BzKsfIpf%2F9348x%2BIp3xrcDcRD9hOrd6J1I3SmW1WJoFZ6Bh1eG4h72MW7gTQ169vRKV4GW21nhThpssvV3xxMuRh5opY9psECAG8jR%2BCOQFdcJM%2FLiM82pPS0DQsyK3EivrRqaaRjDaBp%2BusPVmWcJBZDlGoglcb0fvk0DJOuvt0FqbSOkYfs%2FnVhKD8nWEVa6JPiBERQdkb3jkQVRp%2F2HA1yYP8brW2FkqJgglyLv5g5EStMTlmGtHpzAFX034Vne1233jCF66Vn%2BEPT%2F%2FNSOCjsjyHDoPZ5QBDCYI%2FXjdKDrWauU97kcPzDWwqna0M8Md3c5iCqOjBg%3D%3D)

# So downloading the app is a Rare Event . It is a highly unbalanced datset with only 0.25% of clicks leading to Downloads. We may have to figure out a good sampling technique.

# IP, App, Device, OS and Channel are the categorical variables. Lets look at the distinct count of these categorical variables.

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15413/20487/Screen%20Shot%202018-03-08%20at%2012.08.48%20AM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521022987&Signature=ZHxRjfAF41wTGZEb9KR0bxL7U7BIk3GQwMlDR0xAU%2BZmUAyXF0fHPp6E8%2FdkX7GUwcgWLnzZv%2B%2Bq9cUdpFPH62TGju17R0%2FpgPFv3l%2Fkorppf2Rbp%2BuHbhOvmHZtBKBRMLMaGLiW6LoQ%2BY95ONJt9nHA5zNYLDaf5OsNTPCF09QxpBAS36ULPU%2FDhahi6V0D7psOUtne7fjD7tfXq1OeJd3ONkQ9qZesp5fsC0VzBldYzApZETfEWYdCq8iocXmq2SO1ZaXr6PynqnINyjnyGEZOmuwFJJ%2BUbC38dSi9h%2FhPcXjfDtZwMuCgbQWJe2o9ekLAmRBp0txriFAMS9MKTQ%3D%3D)

# **Organised Bot Driven Fraud**

# **Lets Blackist these IPs**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15527/20612/Screen%20Shot%202018-03-08%20at%204.32.07%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521064321&Signature=gYsQpsOMaw8VCw%2BJ%2BzBb4hHu%2FiLjMj9Yvcu0%2Fv8w1TMBfnAwgT5QX7h5DhBOw2E6168uSKWrFzsIsh3iDCDxm0FvBzdU%2FSTdTpFHLuXxyLebpobBjUO2cwhhzAgmxX9KKUrhfxH6rurSAOx7Dj1LtGBBpFTui%2FLQfPXz5Vce09hcuQWAhbDSCFpxAx%2FANi6nPb%2BdG6X95Q4gpSmakf6fN3ayRooTv2rzzHgc%2B45wEtzwdY1Q91AFI7ZH121hjtXoWte8zgce0Z1V80wkp%2B9mY%2FPp%2BG%2B2Aol%2BtyKqpQcwpxY8ZKUFj6HzTGhWiwF2dIspYVm4Otpf1Dktr0uyiNfeGw%3D%3D)

# This is just a snapshot. That's a insane nuber of clikcs coming from IPs that dont convert to a download in a period of 4 days! 
# 
# **Bold Statement : These IPs should become part of TalkingData's IP blaclist. Please comment if you guys think otherwise.**

# **Cluster Model**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16030/21157/Screen%20Shot%202018-03-11%20at%205.48.01%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521064406&Signature=pFagLdCrXhrZmv1HxFbJk%2F9TiX%2Fos%2FWAzieEdfTljlOVBM0wr2tzqflkElX%2BDFYcZFdzBr%2BqxCDC7hY8AM7ddYy9Z4r1TIGdNxg48QMFfzt3EpcfiLKFz6TCYjYUeDX7DA9UgUuQmKq6ocuE2uxoBQspCmbNe3bI%2FB%2Bc7FMzaeZn7o0wNRAkQUV504Krx%2FLF78qIhAAp6Yn7wHfbP0n%2FCJuIAKS6sFl1O2HG3OKLYLgAAu93yLc8O6PjbGoDopRa%2FIxJf0WoObrwA7luKnGah3AQhIZUquJb1%2FAWHmHpFkQryonxS4yB7H2pwULkggUDVHxf36u2yLw0ioM7AczteQ%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16034/21161/Screen%20Shot%202018-03-11%20at%206.18.37%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521065961&Signature=izzpw%2FQqkXG%2Fz0t4u4Uv91ObUiS3zkgYjahWPX3pfj%2BJbsPu7II1A3LL0%2B%2BRNHGhCV2ueb%2FAhmqrpPSv0MWlsaQQy8LxhGfOvudaYN7xtYfQ3kwBCldqtkWFAujIcgXsw3JJ0tDlDnMY6m1cy6Y1wbG4yg6Y4Ji460MSESFGtW5oYDTx1XSDkQdvWlCMJOn3nt4YtCJBw8mslEYafTnx4qSTC1aJw43PQGXRaEZEH2WEAshzhBNryuL1b%2Fi7XjD9fnULyJ%2BNwaLznX5%2FpldOZ3Eyqu2tSDP3VR1KAEBb8HJYVIs2GaO6IG4iBOjW%2FuSBtx54PJA%2BYAY877HLE7WR6A%3D%3D)

# 

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15527/20612/Screen%20Shot%202018-03-08%20at%204.32.31%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521064323&Signature=JYvmy2575HJU2dYFjOk%2FrnnUhLod62ps6zAsZEcwXenREqaTbboekdqb9LObSiJPAAfCVlIy3kbf61S7UhU2xsCMTETfrG2OlFh9d3QWAYIy4Kj8%2BaBeMqO%2BM%2F5bai4pl0qorKZLXhUfCR99hridEPhshOllEotkTBYjKJpLfqTAE5Vd6fY6A8eNvzbhEwkwtXaert0Vx2brpdfOYJxSpX4551V8hfILpY%2BIwFe898LOgEpL1oIic%2BeB8hAmBA7zYnTdiuOLQ2uPDY0LlQBF5NMZixZ5ZemJ4faoj67zkyXBYji1Je0oMuubdQ3LmYBtwstCpFl%2BcjuxJ686s5KbqQ%3D%3D)

# **Cluster Model**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.20.30%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521066733&Signature=Ubcfxn5dJmD8uIybFh1K7dxxRJHmvmBC%2FfuUWmg0WtdcTFSWG5OEyI2ey4yzVH19Q51XaJt8G6kR3tgJRkugCWewEAYZ4A7X%2FSWIAL1VxBIAjYWSNSKMNKVdrzuyWmS7spxqgyKbDTZVMYTMHOPpKxFjbZWm5cCUY%2FEvKrWJbXbn4iCKbUg9b487ml%2BTtJSBD%2FFo%2FdqBkZLIP12mK0HfPKL0srEEQm7N9FN1gUzUW2pX%2FJfcxdv%2Bv7Ds%2FkTy8UStHorucYM2ezij41hCXfev7hD%2Be5LQy%2BE%2FLi0O1PS5fm3BKK7WO8zBGU21ORzypwaYtyGK6mkAsegDoSqp7G3pPg%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.21.09%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521066737&Signature=HnVtUAhqK4GHEz2LSZI%2BO4%2FSkjAZ6Zvm7mAhhbVSYJlmRSKLZk44zwk889DjENHCvz54EnxeW59HqzabOA00IkwTYlgWftEppMyIvz0PDImRfL5GNzeXNwkzTyW0StA4ch6IxNNPu1upBx25f0nD2AUoNhRroZvzcjfMK8VmkErwZQxH93VyDBJHUfl9HY4huoCfo47Vy4iCRFh11y8zt3re44InGoY6Hsx1DZ1iEZNBgwBwMiiWvj1fmaeuBBU3Aq4MzBKeFzRPSfQ1n5bGw4Z%2F%2BtoXnfsHMyCQT%2Fogh%2BU2mvITu3MWoItr3%2FAniak2UcULy77ivtzez7Bvd%2Fwr6g%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15527/20612/Screen%20Shot%202018-03-08%20at%204.32.50%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521064327&Signature=h0SIlHdPwzhoiOEeb0qOZarZJQy27V7dBMA2EyAuV1V%2Bw37L%2FKaEsJHz7jO9vff8PP5Lepc2OvoAzZOI4Lp6mbv4MdmOMo4uaN6COTOx35fbcIpMuVf3x1FpM%2FiI0PWf7Fbjos8C68h7W0fcIhAfXVR4tr%2F1bBCsx6Lt47zTKaXrsHH3cqSTjAegQYf9nyW%2F0bGk53M96SYuoIxX89CFTDTdsvJoEg1wugNYZHuyqhMIk9MNLjXAkb%2FGWqrXX5ePkdDhSD8i1YkjJMXEeOJbf5WYOv6OV4TFo3Jm07smKS3tK0WpazhJX6K1wbLXQ9pmV50vQxjryK3pJQQ5%2B6jsqA%3D%3D)

# **TalkingData should add Device 5, Device 182, Device 1728 to their Device Blacklist**

# **Cluster Model**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16036/21163/Screen%20Shot%202018-03-11%20at%206.35.12%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521066978&Signature=cbTBkIgFEB0%2Bt3FHYj1Rv3%2F8v7%2Bi2Kr%2FrnDfcnYMq0OmswK35Rqm%2BFIsyPYws58Sq9ctOlTMyySDDijFVbVVYKcXWbzz6tBrir57sXCvjAhiMJisvlyeXz1BtwkxcI%2FhsgjKUGDKsiKC4Gld7iO86P%2FkJPHIDBzjrUIIIfNSf4fOpSmjaRf68RVlI2zu7Y8aamUbDtnQfo7Pgn3lXvHLL%2F3TcI1DCOTveSs9A7TxhxqVxwSU%2BLFQXNLXP3kNCLjIFl4O8ptHIS5gWbhekACFe2RXcUPQ6DgwRVoFDQ%2FyH0yiNySBe5sETXY4Jd3XH5h0jN7u%2FVvY5nj9cOhTfoKlXw%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.22.50%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521067014&Signature=HAiq2Qgsrvsl0%2BIu4srmreoyEpT1u%2FCG2DFKnjZZfCoLH%2F3t4lThsQ%2FBNDYmpyudvfA8zfTfJiKHEXXAocHaSCGv8762qf8IMlkqKbZFOCJdOmhy1K%2BMBUuKx%2FECVps4b73hoArKO0WZ%2FhzjXDS%2BmjNeC%2FRl%2B6lsYQJvUfmpkXS%2FxoqgLw7PYJR5QyzoW0geFA6CTXJPpzjfHSsnyFb6bdivbGlR0%2F%2FP1R2dPJIgocj2IOn5YSOu4SRVpwL3Kdm0YqKuBkBHMlE7R8Ktji1m6FdNcr%2Bav97JBin0tM1vqfrYHX7VvSJh3jhnJxgP6HOE0YzdcfQzyvcWv9tlp6MqEg%3D%3D)

# 

# 

# 

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15527/20612/Screen%20Shot%202018-03-08%20at%204.33.05%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521064329&Signature=mh5fSihhl5d2UTeg8AF5xXVGUy2Mh8zqxqdPMrkdsqcBbZ8YIl7oeAXRZMra%2BwwSsy3MPyOaPOVjJRZRESrD8xiM8uATd4XeRtY651maAKKdfI0Cw9W2aLU8bmMM%2FNt8lORTzqJJAqEZLYCiWPMjKhIoIp3WzWzO%2FrY%2BtdvK1hdzVV7%2ByZBghvHw1T8BM8MvseCT%2BJFcsNGDPr0yluScrfUL4O05j%2BQUXovnTBnsjv1ZOZQVZ%2BHVsmhIDVduDhbWh6JtsD33JtY5IQTUzE45SMa1bzd3MuPw0wYoBHvQTgqY2q4sj5gt4ET7PFfvwd8hO1hn%2BUF5U5yk3y2k1mslnQ%3D%3D)

# **Cluster Model**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.27.57%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521067104&Signature=IAP9X1xZCkDzj6xAwX7SLluOunT0D5TTqT0uEem7Zah4xhKM3cGqFsTF2pguc%2FiLU6rrGg%2FSqrDrz8Q%2B%2BEBKO26Brfob34zlR77c1JUl5VaoCP%2Fm%2BnwkqxQujxzibxcQGryRe3cu%2Fxr1tnRTLD3YSqM5yxkxjWfsfAlrg3LvOup5QPi0ko2zQEqapseKrTz3qgCyLBhpUWlwpyWcQBstVIOfwLs03maFaiQPIKddu8S2Hjfh4FybFiPPNJ81xhrj5L56QuG3%2FhXIsgG83C4zkV35%2BJiFebeekFiW8q7SVDvFbgQqs3xJReoMwzCu66ywyyEJDikXhpdAwrAE3c9JWQ%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.28.14%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521067122&Signature=jQb7RmUXcyOJKJQ2GoFIeXQHkB5aguVz2zyya3iZj5gvv5JbLR1D7NU6gIIcMS%2BKT2hR2q6W28vUHSc8mef7%2BTKh06y7ZXl%2BqThuJYHmFcVhwGhxdcR%2FieiuBD0McUkrdDXxS1KRVNa1UCyp0wk5B1QQRH5aMfThygUXWgb%2BVRWnxmHAraVkrLlTCfBu4s65cjjggYQCJllQ7Ls%2BIynCaxWrtfmqPBvPbEULXHMhtw3CtSzjL1w%2FlqjK8nSS7tVfZE1PVhB4ylKEpb2n2nIWEBTXMlCH36jQXIUEt0NHHAjcvkhrlDcUoJj5bQ4eINTjuF5nSvqshqma7KcNN3Bm3w%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15527/20612/Screen%20Shot%202018-03-08%20at%204.33.15%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521064331&Signature=BoFRGdWebz47r5Vc%2BRhQhwBdPSL%2B4x97KXb5jsJeVG5mWLMC4%2BKuGGzuXmLddoaEqGzdmLGgWDDAmfmr%2BjVxeOVXnbLbTRPuWuqrtKhVyMxx5p25lcZKfBPYt5pOIumJ7VBcFDs0xrr9xifPkJBlVV0Dg2IV0ikYsFNmGJB0B0pnb0KhgukFjDmEsgHeDKWFrtUOvvLRLn3CwQ51%2B%2Fg8OU1XBIqNfIr8FI10ewqwhCluuUieNud1MkZK93KOwaRlOAhQoHSBg67kH7EXPrlOWsgUx8x%2B0nrPBMcGwmcBHlymk034Nkl9HcepM40RnVESUyX%2FPOjfxwRWXMnsIY1zaQ%3D%3D)

# **Cluster Model**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.28.40%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521067184&Signature=Pr9CWoc1EEvzVBhta%2BiTp4DrUxB6B8lZX%2FCoUa1Xp7s7%2FHjcwLGMehq9LwZWwCFs3MRMU0IKr7uFwWk%2B1IkEODkcL%2Fs4dXTfjOH7k5heoL%2Fl4mvLlB%2BUJjw07GGlc%2BxAm4yDWDuLvKi%2FCRSWenjLKsPDdE54%2Bqt%2BfmPn9Fai%2F9%2FwyIdFVX%2FdXI52YS5y%2FbhFiY7KbcOSOeFaf%2BiVCOM49lkjm%2Fq%2BWaiJIO6R5R3HG6S4yIK4fO6796utG9EWDhvhgiL17ToYQ3q7SiiVviRv%2BimsKeD1ZJCk4a20BPqd50H%2FAyBII3bVCl3pi%2BmaA5JlkI%2BUSERyCL1aqOlp3hyzyA%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/16035/21162/Screen%20Shot%202018-03-11%20at%206.29.00%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521067187&Signature=fBDP053LiiQDMZ2a7NbToLpeygCe4ZBN1md38yCdYQo%2Fk2HK1gPVPyk8O%2Bi4a4wuzEwqAwUUnBExE0KEKU%2B7sRUkr7wFTX7iCwItBsUjo77ZN2p5FPeNL%2B1eriPlTfXKeiWVu9TxUgk71vi8qin%2Fkzl8wNsKFi8NWrfi7gNgzW%2FUGV62iaGytI1cTOaQsRZIHf%2BVU5LFudFq2lICC9g3f8BIYv%2Bv5o7Vk2SKolqRUy2%2FanuMDdSZ7p3T6DD438aRP2qFKRuf5EQFL24tZNV0ehDNNqpcaZklBltMnUz%2FZsVFiAm9tGfCDPODUql%2FiQSOs4utMI4jVYfMdjqv5xorhQ%3D%3D)

# 

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15409/20483/Screen%20Shot%202018-03-07%20at%2011.58.17%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521023320&Signature=RYJR%2BcrquTK4klb3xmHBqltL35nE7k9rDf9IsqVa%2FwS1OSn3hgwYrh6tzZ86NI%2B%2FiOesMXBTZeIW9J8x%2FHt%2BuClCAafzUuTZfInD4XT%2B7X8QWCTsN3dpDn1uKgOaowITZvNLduT72qe%2FpG8lvjVumq3YD0fQi6t7Fm7hvyHfBug0OzjMKux9tHk6bnodl50DCjAGZXCnp3DSBn01cds7uaslkQ64C6Ft51OHMxpaCPStedANm3JzkFZ3ihsGrqVIJZ1CgMWKFzKxX3abUVYZA13s6pNXhRB%2BRgKceN3X7shf75RN9PJ3d4BKxqD1g903KZGmkiPgjz9sRR95c7AYrQ%3D%3D)
# 
# Majority  of the Fradulent Clicks are from Device 1 and Device 2

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15406/20480/Screen%20Shot%202018-03-07%20at%2011.52.00%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521023254&Signature=MH64uCJdyNLuTFLojsUNp6jlYZFqidPegQBTQM9e3f1x0HOMvn3JT3sxPF%2F8AmZwdy2c7A3vT1NF67%2BnXk%2BbQlynkEw0CCJTGef0ZoMY2vlLMk3BB5V2EyB2iNMHJS2l%2Bh7weGRejE32HWQC9yta0mUsaoOQ9yxWUj981%2BItNcVsqdIMIOkP7kFfYdk869ZtBeBC4ftp7wTW02txy25CNuQshvmDFRF7ztp9ICcmRZvPVljPdvONlqE%2BN99LlNrgtU8lEbACmiyHSi62scl9QGgaPJOpg6OFEenqBswqjOZOgKkqlARPlq0fFjKabc7Ik783q%2FkpGHlYj28i%2BOYLLA%3D%3D)
# 
# Majority of Non-Fradulent Clicks are from Device 1 ans Device 0
# 

# **Effect of Click Time**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15657/20750/Screen%20Shot%202018-03-09%20at%206.42.09%20AM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1520855030&Signature=PkGlf7SV%2B2Dvi9Jo1pyKKYVloUQe4hzqoDpTX6orebbKi6d%2FvJJbMrgfr9Rt15433KF3X54oAys%2FetG9%2FDuqPyIAw%2BmElkQEavX%2FCThHlfS2n8a9HEKf3kQOfbjGiHdBhKYmq1twQzGOvMiPXe3bCA11bfW359CjLym2huWByUyTmPJZhgrq007%2FCfW%2F76fKXVCCCNHBr%2B8eLrsryEw8opQnPzJ48IMurP%2BdVvAHDuZkdnstaDl44rzpp5UvOLyE8xt4oZi%2B0ySIFbfhhnFDSobeqmXpDWXq92NwPrlOR1nPYlAS0d8DbAjxIHtKTuDXfXWHkGIQSlSXaHD1JUSX7w%3D%3D)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15656/20749/Screen%20Shot%202018-03-09%20at%206.39.09%20AM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1520854878&Signature=gmD0GtHmOMo8D9fFSp6JfOQ1iqKUi%2BpalMwmN9ts%2By3wpk2MwsHxH5dn5h6Ez%2BUhNmdqWMJhckdMluzwunQz%2FXv8O2zCUMWLMVSNNTZWMvfiHyCmw32ldH6yiUX5yw6WYR0MQgqYBtu4uOLoLSr%2FtrQLVP85krVvQsofgZ025Sb28u9MWzoK93puTSMd7mJKp36SJ6BUhz3ibwp7Y0t8kgHxVAoUZ47dfjnLQ6Rs8EyYeJ4liiEMAdYQ4kS4dziHRtv%2BCLusJ04ZZnNAvlKEoCBQk4BJ7knJ4YoF9KhIJDMO5Rc50Pr0wzW0RywJjbzM%2BF9QJWtt4O8qqnxpO%2FPtOw%3D%3D)

# ***Day***

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15416/20490/Screen%20Shot%202018-03-08%20at%2012.38.15%20AM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521023294&Signature=ikwevCBll%2F%2FXeLYp4sbfOKCnt7aQ5nrWflMQ9w3V0qXr1KtFKUq0OoXfbsb46hEHsHEmd66LNuaNYJuB%2B192ORzEiR8oU%2FVWj8L623%2F3R3IQwHuekaIFKSjact9yA%2Bm%2BbKsqNnS5DhSSVVHqaCVD52Oxs7lgopn0t%2FL2W37AVvzDBqXINhz8eZRVRmx5NM695EPeo5LoYtXlu1WWIeMBTM%2BLlabulHXgspwI9nZdgF%2FyFD3xGzTBkZ9DdkjCeEAvNjUJlj23969IGWJJ%2FxDdHUeXrVDVfzYuRafh82%2FKnBwWk6BGWYsb1w3fiJyzE6QQekFjT9sBflJ%2FvdhebBm9gA%3D%3D)

# ***Hour of the Day***

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15359/20432/Screen%20Shot%202018-03-07%20at%203.53.43%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000106&Signature=TfFcHtrQDOq1MXgnkh6MK%2FKw1AFuZTL9P%2BcUOQjUFKcmERCZ8wDu30JXOKRzLdy7bpuZ6BqNMl%2BNxJwIh6irndM036yH8Ae3Jq3ahom4QO32RLsyzHZzmrhvPU3rdtWSapt9qftXszQkAG3XUd%2FJeDiGLzquH9Cz58ItmFavy5l%2FHvLDwAw%2F5nKK7ZxrcsIuDv5SMCwZIdPDe1fhvA%2B1Dkw8CvFiSrRopPtcrzWdCCCzk3nzgEaqbkdwBNr3mXO5hOObgVTIoBr5oT6qN%2FGzpieTw33fipeInXXpoAeIlv9TmEjeIcH1Ce1qmGHUqSkYGSf9bVHdUZ3NnVK6ol%2FOcQ%3D%3D)

# Looks like we can tease out some information by extrating the Day and Hour of the Click Time.

# **Scatter Plots for Fradulent Vs Non-Fradulent Clicks**

# **Channel**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15362/20435/Screen%20Shot%202018-03-07%20at%204.08.35%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000296&Signature=ZgHAcOAHhacBo1CQaCAJ5i7LDVLhqyfv47DHOBSHWuirBQCT6C8FlFOnKgzlHOcxkDRNsRoUGbSBinMf9qfGSCYD6TxW181nHQz8Cyt%2FcF9p9ESBLf0xJUeZqxEC6PNVOUjZJ5VuouQgNJWeypKbrvg%2F15AGhpCmyndxY82C22CUZBJKnUvMQzm154NtqncWLnTMEhCcNoVPlKW9w7EF3nzfXOUlUxIL9pLKEbq2sPiTFwILDE1xeerndLsobOkVo4ULQrsXd4YygQfX1ktT3sN84qlL37%2Ftug0MgJKUe8kiruysYIRgyREjsSrAuXJOmxB%2BDOaouYpVlon%2BL9lwnQ%3D%3D)

# R - Aquared = 0.001
# 
# p - value =0.54
# 
# **Many Channels have unusually high number of Fradulent Clicks than Non-Fradulent Clicks**

# 

# ![](http://)

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15360/20433/Screen%20Shot%202018-03-07%20at%204.03.44%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000293&Signature=XoQUI5XtbbWVI5ZN64q%2FM6e12sqmB%2F40tZZC44xIZZlEkeeRDCBuVa0RN%2BdvHsoCgXeNeb0iMFrW2YraAVN35wPWNNyXVjXnMlOixhHhMrtLRAQ9eOgvAdZH0dyI2CTI6Lm3CvCH821YmAQ7Yl2ZqLFFOLevBn0b2sJLjuIFa2VFuWyGynQOTCgNBbpY7ePkrEh4siykuf%2Fn8JgkrKvhDDr3ghHl8a5wy43rWdY3IFYRAQMr3gcLmvlH%2BMMhmboht5vZX6m0Fv18vpN0AxkPqbqAG4aI8ol7kRjYCnC4g%2FaQyZJtrFKyMvAI8hjD4CBFOTU6MCNCyJSu%2BNLmC62D2Q%3D%3D)

# R - Aquared = 0.86
# 
# p - value = < 0.0001

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15363/20436/Screen%20Shot%202018-03-07%20at%204.32.09%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000289&Signature=idkj1KBBcQQZKaRY5JWmjIKzeZkeWdT%2FiRnvmQvhZRaINvEiEqXmIhF3WSpiia5cOY8mRdGgU2E36FWn1p9N1UqSP1jP8PJhBUVgG2Aj3fNkFnfx1qp%2FH%2FJUwNA%2FbvG27Xrwv%2FoLODoMBeDVaieD0e4PgYbsuof5Y2xom1W6%2BXnobmc5bK%2F3J3w6GjDunNJKI9Cg%2BZ6dLBpA1E0im1k6MursBixUiAoicCNv51AdGIRMeuBIL1kQac%2BqjxXJX880h90d2ZdfblIyXgls2lY0XApngqwJPaKiyo4ewwrJZAsnFV4LUrMBcVsAJizU0t7cZA5i6BpkBEGHm6cHsxF12w%3D%3D)

# R - Aquared = 0.79
# 
# p - value = < 0.0001

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15364/20437/Screen%20Shot%202018-03-07%20at%204.37.26%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000304&Signature=AUyruML7FxWdFLZAWC9n9LqV2XixUl0Xn8uqY0D0O3E%2BoisP6yhNetNlykHaf7zu%2F0NC1hLwbT77HQKV6HWJ9swcVspfbg0wTyvSXtDC3QI3RUVVgFrEO4ELi%2BZ1U4ML3pZ2UqF9Z5mKCLclUrvo27d06Yi9qr3iy3Trz6ZNrh8xiiIvss25mga7zYSkAhXm%2FalECnqGDI6PCU4l5wBdvcM0uiGIEsX9fkaUmtMk%2BWHo3RUtg4a8g2imF%2FFTf%2FI0Vy4nqyEIpQcrDfrLmUqM8yG1ukhFAOhg3gtfWhOBLr7D0CN4ZeScU1VQbzQOgcrTeSpvQ%2BuElleROfKhJ%2BhsKQ%3D%3D)

# R - Aquared = 0.90
# 
# p - value = < 0.0001

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15366/20439/Screen%20Shot%202018-03-07%20at%204.42.03%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000308&Signature=mwfx1r5S653LP74G0tHgxvS9I7DoyqBtIFrfk8c%2Bl9uffb6MX3MQ1UcLEkL2ZhjeNMQtyKF3Dl%2BCDX7ClgPtVG1tj8JuFRbJk3n0UdmVx5tocHertHSqN3Gf0X2XoNpdMBzUcTK9sSO3QUJ9E82T8bzDNWVxDoSZv0f7t7hSaGqVEX1hXpBe52lCD9%2FtmpIQoeWrHZnA0cblA0IdZXv58Gfv2kSuRePeiEwYn1XnuWeO7DJIZgFSawfwWHGQq3DAO43ivMp1ioinH2Ne%2FRm5nwa6rLWNwQuDdd2heW1VLWLdycAbB2IXn5p6VU9zv6X7SSh0fv%2BIkTUWj754KS1O0w%3D%3D)

# R - Aquared = 0.01
# 
# p - value = < =0.00006
# 
# **Many Apps have unusually high number of Fradulent Clicks than Non-Fradulent Clicks**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15368/20441/Screen%20Shot%202018-03-07%20at%204.52.47%20PM.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521000301&Signature=h5oWc5do0%2FCEl%2F0oRimsJpjtyondNWtXizogRzdF3dH98mHrnpwlnSjAYd9U0%2FZusV9am7fl9fLZynQ0OHk1VlxYA0M8BH723O3X3tqo60SyxyuTfGjfKTPFGw%2F01%2BvDltNTLjtSd%2BiRrRHBHd9nutJfiwm%2F5rbw5DnpNKjqTl%2FYr1j5jAiIm%2BJ41y7Crk4iK0pHGNtDe%2FpTo11LVNLryzykY43QSZ4h6yKukUsmCujwne6dXZd2uhxChkivTKiu4ZImQwoCyD7dpC3JVui8erAYo85a6uDic78tz3mpLDJrKlXwQOh8QFtGrxvgayNGuiY6TegoyQvx12IwCQ4ZKw%3D%3D)

# R - Aquared = 0.68
# 
# p - value = < 0.0001

# **The below one is for fun...**

# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15405/20479/abc1xyz.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521023155&Signature=ARrLNXrJHjn8%2F7jjFeiYiBDgnaTqz9dWFH4d7VmowFYtuEE5CXcSJbByGq88aVX0Ww%2FdyL5n0BaKf91v7K7aENkRYaFads%2FrhD1yVEyytK4HypFLroFN9UJAES7Ki0O7pHKz8xx9Lc%2BL8aIiTl9X2c1GWPMTLRi%2BT4UUcLivIm8bG%2FWZYLxFlUVN6pBk8lrrcBEHtxGuQxaZ8PsCKD4MN6U%2B0REKVIYz8C2wgzJLuU3j%2BLWGsD8tzgvguGhTMiDUKreyjvaD5F1T2xIWrIwmM01GJFZue8oTBX88RzJUxDjti9c72KVP7PIlCGPkMg247vCkhW8kOC5AhenEZmvV%2Fg%3D%3D)
# ![](http://storage.googleapis.com/kagglesdsdata/datasets/15405/20479/abc2xyz.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521023188&Signature=AYYliz8ff4SxM%2FLbNpJfD%2FdWRpqB4kFNPDTQxDyx4upbLmFgKnLuikOGA7S5XNSnCqk2OTVqXoCN%2BHFoI%2FYPXt6s3WAoKi7W7162Tuhd0iVLoqteCxJEGwCljsI8DmbUOD01UQqQgAc1eQgK8cdQMA64t9DIYyXkGJBaQ4V%2BmoXFX1ByNVb5EKtLEXSu2ThIAU9wqVOfKjDTUZD%2Fz0bb3AHesltluO6uHnsnZaGuhGDeyh8NikshtFIca5R5Bk19DemiwjATD%2BjnWp7DKH4663LpR6EKvDTgrorcQSVSD5CdWj0ytUrRFFvZvQIMx0y8umaSZZ7a83MzhQnOVltPPQ%3D%3D)

# 

# **To be continued ...........**

# 

# 
