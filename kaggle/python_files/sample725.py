#!/usr/bin/env python
# coding: utf-8

# Following is an aggregation of the details of most of the columns in the training data. The details were obtained by googling.
# - **MachineIdentifier** - Individual machine ID. Training dataset contains 89,21,483 (8 mn) unique machines.
# 
# - **ProductName** - Defender state information e.g. win8defender
# 
#   ```
#   win8defender     8826520
#   mse                94873
#   mseprerelease         53
#   scep                  22
#   windowsintune          8
#   fep                    7
#   ```
# 
# - **EngineVersion** - Defender state information e.g. 1.1.12603.0. There are 70 unique Engine versions in the training data. The 5 most common ones are :
# 
#   ```
#   1.1.15200.1    3845067
#   1.1.15100.1    3675915
#   1.1.15000.2     265218
#   1.1.14901.4     212408
#   1.1.14600.4     160585
#   ```
# 
# - **AppVersion** - Defender state information e.g. 4.9.10586.0. This has a long tailed frequency distribution, with a 110 unique App versions. Most common 5 versions are :
# 
#   ```
#   4.18.1807.18075     5139224
#   4.18.1806.18062      850929
#   4.12.16299.15        359871
#   4.10.209.0           272455
#   4.13.17134.1         257270
#   ```
# 
# - **AvSigVersion** - Defender state information e.g. 1.217.1014.0. Has 8,531 unique values.
# 
# - **IsBeta** - Defender state information e.g. false
# 
#   ```
#   IsBeta  HasDetections
#   0       0                4462557
#           1                4458859
#   1       0                     34
#           1                     33
#   ```
# 
# - **RtpStateBitfield** - (Most likely - RTP state: <Realtime protection state> (Enabled or Disabled)
#      [source](https://docs.microsoft.com/en-us/windows/security/threat-protection/windows-defender-antivirus/troubleshoot-windows-defender-antivirus). Expected binary values, don't understand why values are multiple integers though. Frequency table - 
# 
#   ```
#    7.0     8651487
#    0.0      190701
#   NaN        32318
#    8.0       21974
#    5.0       20328
#    3.0        3029
#    1.0        1625
#    35.0         21
#   ```
# 
# - **IsSxsPassiveMode** - Google searches suggest that this a active/passive mode of operation for Windows Defender. If another third party primary antivirus exists on the system, the Defender enters Passive mode. Passive mode obviously offers reduced functionality.[Source](http://techgenix.com/stick-with-windows-defender/)
# 
#   ```
#   0    8766840
#   1     154643
#   ```
# 
#   Data shows that a machine which is in passive mode, has a higher prevalence of malwares.
# 
#   ```
#   IsSxsPassiveMode  HasDetections
#   0                 1                4402017
#                     0                4364823
#   1                 0                  97768
#                     1                  56875
#   ```
# 
# - **DefaultBrowsersIdentifier** - ID for the machine's default browser. [This column has 2017 unique values, so these couldn't be direct browser name mappings. So probably, a browser-version combination ? Anyhow, this is what the top ten in the frequency chart look like - 
# 
#   ```python
#   239.0     46056
#   3195.0    42692
#   1632.0    28751
#   3176.0    24220
#   146.0     20756
#   1910.0    19416
#   1727.0    17393
#   2064.0    13990
#   2725.0    13338
#   1160.0    12594
#   ```
# 
# - **AVProductStatesIdentifier** - ID for the specific configuration of a user's antivirus software. 28,970 unique float values.
# 
# - **AVProductsInstalled** - Assuming this to be the number of anti-virus products installed. 90% of the machines have 1-2 products installed. We should probably drop the one row with 0 AVProductsInstalled. Again, "probably".
# 
#   ```
#    1.0    6208893
#    2.0    2459008
#    3.0     208103
#   NaN       36221
#    4.0       8757
#    5.0        471
#    6.0         28
#    7.0          1
#    0.0          1
#   ```
# 
# - AVProductsEnabled - NA
# 
#   ```
#    1.0    8654101
#    2.0     198652
#   NaN       36221
#    0.0      25958
#    3.0       6075
#    4.0        453
#    5.0         23
#   ```
# 
# - **HasTpm** - True if machine has tpm. A Trusted Platform Module (**TPM**) is a specialized chip on an endpoint device that stores RSA encryption keys specific to the host system for hardware authentication. Each **TPM** chip contains an RSA key pair called the Endorsement Key (EK). The pair is maintained inside the chip and cannot be accessed by software.[source - google search]
# 
#   ```
#   1    8814167
#   0     107316
#   ```
# 
# - **CountryIdentifier** - ID for the country the machine is located in. This has 222 unique int64 IDs. Wikipedia cites 255+ countries and independent territories. [source](https://en.wikipedia.org/wiki/List_of_countries_by_United_Nations_geoscheme)
# 
#   If these are exact country codes, then Austria (43) has the highest number of rows in this data set, while USA(001) has just 2 %. 
# 
# - **CityIdentifier** - ID for the city the machine is located in. 1,07,366 unique cities and huge number(~5%) of NaNs.
# 
# - **OrganizationIdentifier** - ID for the organization the machine belongs in, organization ID is mapped to both specific companies and broad industries. There are 49 unique organisations, 50% of the computers being under one org, another 25% not-classified. Here's a breakup of the top 5 values - 
# 
#   ```
#   27.0    4196457
#   NaN      2751518
#    18.0    1764175
#    48.0      63845
#    50.0      45502
#   ```
# 
# - **GeoNameIdentifier** - ID column for the 292 geographic regions, a machine is located in.
# 
# - **LocaleEnglishNameIdentifier** - English name of Locale ID of the current user. The column contains 276 locale int64 IDs. "A locale is neither a language nor a country, the same language may be spoken in multiple countries (often with subtle differences) and a single country may speak multiple languages. A locale is therefore an area where a particular language is spoken which may (or may not) align with geographical and/or political boundaries." [source](https://ss64.com/locale.html)
# 
# - **Platform** - Calculates platform name (of OS related properties and processor property). Frequency table -
# 
#   ```
#   windows10      8618715
#   windows8        194508
#   windows7         93889
#   windows2016      14371
#   ```
# 
# - **Processor** - This is the process architecture of the installed operating system. Frequency - 
# 
#   ```
#   x64      8105435
#   x86       815702
#   arm64        346
#   ```
# 
# - **OsVer** - Version of the current operating system. 
# 
#   ```
#   10.0.0.0        8632545
#   6.3.0.0          194447
#   6.1.1.0           93268
#   6.1.0.0             582
#   10.0.3.0            225
#   10.0.1.0            141
#   ```
# 
# - **OsBuild** - Build of the current operating system. Has 76 unique build numbers, of which ~5 form the majority. Distribution of the top 10 values - 
# 
#   ```
#   17134    3915521
#   16299    2503681
#   15063     780270
#   14393     730819
#   10586     411606
#   10240     270192
#   9600      194508
#   7601       93306
#   17692       3184
#   ```
# 
# - **OsSuite** - Product suite mask for the current operating system. This has a very skewed distribution.![img](data:image/png;base64,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%0A)
# 
# - **OsPlatformSubRelease** - Returns the OS Platform sub-release (Windows Vista, Windows 7, Windows 8, TH1, TH2). Less skewed distribution than OsSuite.
# 
#   ```
#   rs4           3915526
#   rs3           2503681
#   rs2            780270
#   rs1            730819
#   th2            411606
#   th1            270192
#   windows8.1     194508
#   windows7        93889
#   prers5          20992
#   ```
# 
# - **OsBuildLab** - Build lab that generated the current OS. Example: 9600.17630.amd64fre.winblue_r7.150109-2022. Top 5 values by counts- 
# 
#   ```
#   17134.1.amd64fre.rs4_release.180410-1804                     3658199
#   16299.431.amd64fre.rs3_release_svc_escrow.180502-1908        1252674
#   16299.15.amd64fre.rs3_release.170928-1534                     961060
#   15063.0.amd64fre.rs2_release.170317-1834                      718033
#   17134.1.x86fre.rs4_release.180410-1804                        257074
#   ```
# 
# - **SkuEdition** - The goal of this feature is to use the Product Type defined in the MSDN to map to a 'SKU-Edition' name that is useful in population reporting. The valid Product Type are defined in %sdxroot%\data\windowseditions.xml. This API has been used since Vista and Server 2008, so there are many Product Types that do not apply to Windows 10. The 'SKU-Edition' is a string value that is in one of three classes of results. The design must hand each class.
# 
#   ```
#   Home               5514341
#   Pro                3224164
#   Invalid              78054
#   Education            40694
#   Enterprise           34357
#   Enterprise LTSB      20702
#   Cloud                 5589
#   Server                3582
#   ```
# 
# - **IsProtected** - This is a calculated field derived from the Spynet Report's AV Products field. Returns: a. TRUE if there is at least one active and up-to-date antivirus product running on this machine. b. FALSE if there is no active AV product on this machine, or if the AV is active, but is not receiving the latest updates. c. null if there are no Anti Virus Products in the report. Returns: Whether a machine is protected.
# 
#   [jk: A machine that is not protected by an anti-virus has a lower chance of infection than one that is protected.] 
# 
#   ```
#   IsProtected  HasDetections
#   0.0          0                 298904
#                1                 184253
#   1.0          1                4261098
#                0                4141184
#   ```
# 
# - **AutoSampleOptIn** - This is the SubmitSamplesConsent value passed in from the service, available on CAMP 9+. [No clue what this is.]
# 
#   ```
#   0    8921225
#   1        258
#   ```
# 
# - **PuaMode** - Pua Enabled mode from the service. "The Potentially Unwanted Applications (PUA) protection feature in Windows Defender Antivirus can identify and block PUAs from downloading and installing on endpoints in your network.
# 
#   These applications are not considered viruses, malware, or other types of threats, but might perform actions on endpoints that adversely affect their performance or use. PUA can also refer to applications that are considered to have a poor reputation." [source](https://www.tenforums.com/tutorials/32236-enable-disable-windows-defender-pua-protection-windows-10-a.html)
# 
# - **SMode** - This field is set to true when the device is known to be in 'S Mode', as in, Windows 10 S mode, where only Microsoft Store apps can be installed
# 
#   ```
#    0.0    8379843
#   NaN      537759
#    1.0       3881
#   ```
# 
# - **IeVerIdentifier** -  Retrieves which version of Internet Explorer is running on this device.[source](https://docs.microsoft.com/en-us/windows/privacy/basic-level-windows-diagnostic-events-and-fields-1703#census-events) This has 303 unique values. Here are the most frequent values, uptil a NaN.
# 
#   ```
#   137.0    3885842
#    117.0    1767931
#    108.0     474390
#    111.0     467828
#    98.0      354411
#    135.0     217458
#    53.0      204952
#    74.0      202542
#    94.0      173593
#    105.0     173448
#    333.0     156391
#    107.0     128633
#    103.0     114952
#    96.0       83559
#   NaN         58894
#   ```
# 
# - **SmartScreen** - This is the SmartScreen enabled string value from registry. This is obtained by checking in order, HKLM\SOFTWARE\Policies\Microsoft\Windows\System\SmartScreenEnabled and HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SmartScreenEnabled. If the value exists but is blank, the value "ExistsNotSet" is sent in telemetry.
# 
#   Windows Defender SmartScreen helps to protect your employees if they try to visit sites previously reported as phishing or malware websites, or if an employee tries to download potentially malicious files.
#   This only applies to Win 10 and Win 10 mobile.
# 
# - **Firewall** - This attribute is true (1) for Windows 8.1 and above if windows firewall is enabled, as reported by the service.
# 
#   ```
#   1.0     8641014
#   0.0      189119
#   NaN       91350
#   ```
# 
# - **UacLuaenable** - This attribute reports whether or not the "administrator in Admin Approval Mode" user type is disabled or enabled in UAC. The value reported is obtained by reading the regkey HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System\EnableLUA.
#   UAC stands for User Access Control and here is an interesting discussion on its usage & its annoyances. [discussion](https://www.reddit.com/r/Windows10/comments/8dxzcz/uac_do_you_want_to_allow_this_app_to_make_changes/)
# 
# - **Census_MDC2FormFactor** - A grouping based on a combination of Device Census level hardware characteristics. The logic used to define Form Factor is rooted in business and industry standards and aligns with how people think about their device. (Examples: Smartphone, Small Tablet, All in One, Convertible...)
# 
# - **Census_DeviceFamily** - AKA DeviceClass. Indicates the type of device that an edition of the OS is intended for. 
#   This has a very high proportion of Windows Desktop and very few of Windows Server and Windows.
# 
# - Census_OEMNameIdentifier - NA
# 
# - Census_OEMModelIdentifier - NA
# 
# - **Census_ProcessorCoreCount** - Number of logical cores in the processor. Four core processors are the most common. There are 45 unique values in total:
# 
#   ```
#    4.0      5430193
#    2.0      2311969
#    8.0       865004
#    12.0       92702
#    1.0        70390
#    6.0        69910
#   NaN         41306
#    16.0       18551
#    3.0        13580
#    32.0        2136
#   ```
# 
# - **Census_ProcessorManufacturerIdentifier** - NA. Frequency distribution - 
# 
#   ```
#    5.0     7839318
#    1.0     1040292
#   NaN        41313
#    10.0        339
#    3.0         218
#    9.0           1
#    7.0           1
#    4.0           1
#   ```
# 
# - **Census_ProcessorModelIdentifier** - NA. 3,428 unique values
# 
# - **Census_ProcessorClass** - A classification of processors into high/medium/low. Initially used for Pricing Level SKU. No longer maintained and updated
# 
#   This column is mostly empty.
# 
# - **Census_PrimaryDiskTotalCapacity** - Amount of disk space on primary disk of the machine in MB. Most popular disk sizes are 500gb and 1Tb, in that order. Unique value count: 5,735.
# 
# - **Census_PrimaryDiskTypeName** - Friendly name of Primary Disk Type - HDD or SSD. HDD is twice as popular as SSD, in this dataset.
# 
#   ```
#   HDD            5806804
#   SSD            2466808
#   UNKNOWN         358251
#   Unspecified     276776
#   NaN              12844
#   ```
# 
# - **Census_SystemVolumeTotalCapacity** - The size of the partition that the System volume is installed on in MB. 5,36,848 unique system volume sizes exist. The following frequency count was created after converting the data into Gigabytes and then binning them. 
# 
#   ```
#   a. <= 50          689171
#   b. 50-100         960770
#   c. 100 - 250     2845915
#   d. 250 - 500     2695227
#   e. 500 - 1000    1601322
#   f. > 1Tb           76014
#   ```
# 
#   Source code for creating the above cuts:
# 
#   `_ = (round(df[col]/1032, 0))
#   pd.cut(_, 
#   ​      labels=['a. <= 50', 'b. 50-100', 'c. 100 - 250', 'd. 250 - 500', 'e. 500 - 1000', 'f. > 1Tb'],
#   ​      bins=[0, 50, 100, 250, 500, 1000, 10**4]).value_counts().sort_index()`
# 
# - **Census_HasOpticalDiskDrive** - True indicates that the machine has an optical disk drive (CD/DVD). Most systems don't have a disk drive. This column could provide an idea of how old the chassis is. And, older chassis might correlate with higher malware rates. Interestingly, none of the machines have a disk drive.
# 
#   ```
#   0.0    8921483
#   Name: Census_HasOpticalDiskDrive, dtype: int64
#   ```
# 
# - **Census_TotalPhysicalRAM** - Retrieves the physical RAM in MB. Since these are in MB, these are converted to GB by dividing them with 1032 and then the frequency count is created:
# 
#   ```
#    4.0       4102156
#    8.0       2200604
#    2.0       1108274
#    16.0       532480
#    6.0        400223
#    12.0       160345
#    3.0        156306
#   NaN          80533
#   ```
# 
#   Most popular - 4 Gb, 8 Gb, 2 Gb and 16Gb.
# 
#   Source code: `(round(df['Census_TotalPhysicalRAM']/1032, 0)).value_counts(dropna=False).sort_values(ascending=False)`
# 
# - **Census_ChassisTypeName** - Retrieves a numeric representation of what type of chassis the machine has. A value of 0 means xx. Not all values are numerical in this column and the most popular ones don't seem to be mutually exclusive. For example, in the top5 common values, both `Notebook`  and `Portable` could be used interchangeably.
# 
#   ```
#   Notebook               5248812
#   Desktop                1872125
#   Laptop                  685581
#   Portable                360903
#   AllinOne                204295
#   ```
# 
# - **Census_InternalPrimaryDiagonalDisplaySizeInInches** - Retrieves the physical diagonal length in inches of the primary display. Contains 785 unique display sizes. These can be used together with`Census_ChassisTypeName` to differentiate the devices further.
# 
# - **Census_InternalPrimaryDisplayResolutionHorizontal** - Retrieves the number of pixels in the horizontal direction of the internal display. 2,180 unique values are present. Most popular ones are 1,366 and 1,920 resolution. These values might indicate the age of the device. Higher resolutions most likely mean that the devices are relatively new, compared to the rest. This hypothesis would also be affected by `Census_InternalPrimaryDiagonalDisplaySizeInInches`. Most likely, larger displays will have higher resolution.
# 
# - **Census_InternalPrimaryDisplayResolutionVertical** - Retrieves the number of pixels in the vertical direction of the internal display. 1,560 unique values.
# 
# - **Census_PowerPlatformRoleName** - Indicates the OEM preferred power management profile. This value helps identify the basic form factor of the device
# 
# - **Census_InternalBatteryType** - NA. Has a lot of inconsistent naming schemes.  For example - '#', 'lion', '4cel', 'l&#TAB#'. Majority of the machines are still represented in less than 10 labels. Some of these seem similar, for example - lion, li-i and liio could possibly be placeholders for lithium-ion batteries.
# 
# - **Census_InternalBatteryNumberOfCharges** - Assuming this to be the number of battery cycles. If battery cycles are set to zero, could it be that these devices were in the first cycle of battery charge / are VMs or desktops ? What makes it more interesting is that 56% of the machines are in their first cycle of battery charge OR are non-battery operated.
# 
# - **Census_OSVersion** - Numeric OS version Example - 10.0.10130.0. Contains 469 unique values.
# 
# - **Census_OSArchitecture** - Architecture on which the OS is based. Derived from OSVersionFull. 
# 
#   ```
#   amd64    8105885
#   x86       815252
#   arm64        346
#   ```
# 
# - **Census_OSBranch** - Branch of the OS extracted from the OsVersionFull. 32 unique values. Five most common values: 
# 
#   ```
#   rs4_release                  4009158
#   rs3_release                  1237321
#   rs3_release_svc_escrow       1199767
#   rs2_release                   797066
#   rs1_release                   785534
#   ```
# 
# - **Census_OSBuildNumber** - OS Build number extracted from the OsVersionFull. Example - OsBuildNumber = 10512 or 10240. 165 unique values.
# 
# - **Census_OSBuildRevision** - OS Build revision extracted from the OsVersionFull. Example - OsBuildRevision = 1000 or 16458. 285 unique values.
# 
# - **Census_OSEdition** - Edition of the current OS. Sourced from HKLM\Software\Microsoft\Windows NT\CurrentVersion@EditionID in registry. 33 unique values. This column may also be linked to license type(`Census_ActivationChannel`). Top 5:
# 
#   ```
#   Core                           3469991
#   Professional                   3130566
#   CoreSingleLanguage             1945461
#   CoreCountrySpecific             166100
#   ProfessionalEducation            56698
#   ```
# 
# - **Census_OSSkuName** - OS edition friendly name (currently Windows only). 30 unique values. Highly skewed by high numbers of:  'CORE', 'PROFESSIONAL', 'CORE_SINGLELANGUAGE' & 'CORE_COUNTRYSPECIFIC'
# 
# - **Census_OSInstallTypeName** - Friendly description of what install was used on the machine i.e. clean
# 
#   ```
#   UUPUpgrade        2608037
#   IBSClean          1650733
#   Update            1593308
#   Upgrade           1251559
#   Other              840121
#   Reset              649201
#   Refresh            205842
#   Clean               69073
#   CleanPCRefresh      53609
#   ```
# 
# - **Census_OSInstallLanguageIdentifier** - NA. 39 unique values.
# 
# - **Census_OSUILocaleIdentifier** - NA. 147 unique values. 31, 34 and 30 are the top 3 Locale indentifiers.
# 
# - **Census_OSWUAutoUpdateOptionsName** - Friendly name of the WindowsUpdate auto-update settings on the machine.
# 
#   ```
#   FullAuto                                 3954497
#   UNKNOWN                                  2519925
#   Notify                                   2034254
#   AutoInstallAndRebootAtMaintenanceTime     371475
#   Off                                        26961
#   DownloadNotify                             14371
#   ```
# 
# - **Census_IsPortableOperatingSystem** - Indicates whether OS is booted up and running via Windows-To-Go on a USB stick. Assuming, 0 to be a "False" for Portability
# 
#   ```
#   0    8916619
#   1       4864
#   ```
# 
# - **Census_GenuineStateName** - Friendly name of OSGenuineStateID. 0 = Genuine. Expected integer values thanks to the description, but looks like we won't need to encode it.
# 
#   ```
#   IS_GENUINE         7877597
#   INVALID_LICENSE     801692
#   OFFLINE             228366
#   UNKNOWN              13826
#   TAMPERED                 2
#   ```
# 
# - **Census_ActivationChannel** - Retail license key or Volume license key for a machine.
# 
#   ```
#   Retail            4727589
#   OEM:DM            3413350
#   Volume:GVLK        450954
#   OEM:NONSLP         317980
#   Volume:MAK           8028
#   Retail:TB:Eval       3582
#   ```
# 
#   - GVLK (group volume license key) is just an acronym for a volume license version of Windows, It's basically a bulk/volume license for Windows. Mostly available to large enterprises and govt institutions and comes with premium support. GLVK also needs a KMS (key management service) for deployment.[source](https://www.cnet.com/forums/discussions/gvlk-windows-8-1-questions-627166/)
# 
#   - KMS Client and Volume MAK product keys, are volume license keys that are not-for-resale.  They are issued by organizations for use on client computers associated in some way with the organization.  Volume license keys may not be transferred with the computer if the computer changes ownership.  
# 
#   - OEM SLP and COA SLP product keys, are issued by large computer manufacturers and use SLP (System Locked Pre-installation) technology to bind the license to the original motherboard via the BIOS and software. The OEM SLP keys self-activate if the corresponding data in the BIOS is correct.  OEM SLP keys, which the user can read in the MGADiag report or software like *KeyFinder*, cannot be used by the end user to manually activate Windows.  The COA SLP key is printed on a sticker affixed to the side of the computer case (desktops), or on the bottom of the case (laptops), or in the battery compartment (newer laptops).  This is the key for the user to enter manually should he need to activate Windows himself.
# 
#   - OEM System Builder**, product keys are for use by smaller system builders, computer shops, consultants, and others who provide computers and services to their customers.  A system builder is defined by the System Builder license as "an original equipment manufacturer, an assembler, a refurbisher, or a software pre-installer that sells the Customer System(s) to a third party." 
# 
#   - Retail, product keys are what the customer gets when he buys a Full Packaged Product (FPP), commonly known as a "boxed copy", of Windows from a retail merchant or purchases Windows online from the Microsoft Store. 
# 
#   - OEM NON-SLP then means license keys from an OEM but not bound to a System Locked Pre-installation.
# 
#     OEM keys are not-for-resale and may not be transferred to another computer.  They may, however, be transferred with the computer if the computer is transferred to new ownership.
# 
#   ​         [Source](https://social.microsoft.com/Forums/en-US/44179f86-f8a6-4dc2-8692-b1637e72280b/windows-license-types-explained?forum=genuinewindows7)
# 
# - **Census_IsFlightingInternal** - 'Flighting' in Windows Defender context means making new development features available as soon as possible, during the development cycle. This does not refer to a public release. The 'internal' most likely means the Window Insider community. [source](https://docs.microsoft.com/en-us/windows/deployment/update/waas-overview)
# 
#   ```
#   NaN     7408759
#   0.0     1512703
#   1.0          21
#   ```
# 
# - **Census_IsFlightsDisabled** - Indicates if the machine is participating in flighting.
# 
#   ```
#   0.0     8760872
#   NaN      160523
#   1.0          88
#   ```
# 
#   Assuming, 0 would mean that majority of the devices have 'flighting' enabled. Which then means, most of these machines are from the Windows Insider program.
# 
# - **Census_FlightRing** - The ring that the device user would like to receive flights for. This might be different from the ring of the OS which is currently installed if the user changes the ring after getting a flight from a different ring. 
# 
#   ```
#   Retail      8355679
#   NOT_SET      287803
#   Unknown      243438
#   WIS           10648
#   WIF           10322
#   RP             9860
#   Disabled       3722
#   OSG               7
#   Canary            3
#   Invalid           1
#   ```
# 
# - **Census_ThresholdOptIn** - NA
# 
#   ```
#   NaN     5667325
#    0.0    3253342
#    1.0        816
#   ```
# 
# - **Census_FirmwareManufacturerIdentifier** - NA. 712 unique values, with majority concentrated in the top 5 firmware manufacturer identifiers.
# 
# - **Census_FirmwareVersionIdentifier** - NA. Around 50K unique values.
# 
# - **Census_IsSecureBootEnabled** - Indicates if Secure Boot mode is enabled. (from google)Microsoft Secure Boot is a component of Microsoft's Windows 8 operating system that relies on the UEFI specification's secure boot functionality to help prevent malicious software applications and "unauthorized" operating systems from loading during the system start-up process. There is a 50-50 breakup of samples based on this criteria:
# 
#   ```
#   0    4585438
#   1    4336045
#   ```
# 
#   Each of these categories, has a 50-50 % breakup of malware classifications.
# 
# - **Census_IsWIMBootEnabled** - NA. WimBoot is an alternative way for OEMs to deploy Windows. A WimBoot deployment boots and runs Windows directly out of a compressed Windows Image File (WIM). This WIM file is immutable, and access to it is managed by a new file system filter driver (WoF.sys).
#   The result is a significant reduction in the disk space required to install Windows. [source](https://docs.microsoft.com/en-us/windows/desktop/w8cookbook/windows-image-file-boot--wimboot-)
# 
#   ```
#   NaN     5659703
#    0.0    3261779
#    1.0          1
#   ```
# 
# - **Census_IsVirtualDevice** - Identifies a Virtual Machine (machine learning model). It is self explaratory except for the phrase:"(machine learning model)"
# 
#   ```
#    0.0    8842840
#    1.0      62690
#   NaN       15953
#   ```
# 
# - **Census_IsTouchEnabled** - Is this a touch device ? Most are not.
# 
#   ```
#   0    7801452
#   1    1120031
#   ```
# 
# - **Census_IsPenCapable** - Is the device capable of pen input ? Most are not.
# 
#   ```
#   0    8581834
#   1     339649
#   ```
# 
# - **Census_IsAlwaysOnAlwaysConnectedCapable** - Retreives information about whether the battery enables the device to be `AlwaysOnAlwaysConnected` .To specify whether Wi-Fi should remain on when the screen times out, set `AlwaysOnAlwaysConnected` to one of the following values [source](https://msdn.microsoft.com/en-us/library/windows/hardware/mt138401(v=vs.85).aspx):
# States:
# 1. 0 or 'Disabled': Disables Wi-Fi from always being on when the screen times out. The **Keep Wi-Fi on when the screen times out** in the **Settings** > **Wi-Fi** > **manage** screen is turned off.
# 2. 1 or 'Enabled': Enables Wi-Fi to always be on by default when the screen times out. The **Keep Wi-Fi on when the screen times out** in the **Settings** > **Wi-Fi** > **manage** screen is turned on.
# 
#   ```
#    0.0    8341972
#    1.0     508168
#   NaN       71343
#   ```
# 
# - **Wdft_IsGamer** - Indicates whether the device is a gamer device or not based on its hardware combination.
# 
# - ```
#    0.0    6174143
#    1.0    2443889
#   NaN      303451
#   ```
# 
# - **Wdft_RegionIdentifier** - NA. 15 unique values. Microsoft documentation mentions regional identifier 

# In[ ]:



