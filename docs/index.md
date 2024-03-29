# ADAFEST: A data-driven apparatus for estimating software testability

**[Morteza Zakeri](https://m-zakeri.github.io)**†

† Ph.D. Student, [Iran University of Science and Technology](http://iust.ac.ir/en), Tehran, Iran (m-zakeri@live.com).

Version 1.0.0 (18 August 2022) ├ Download [PDF] version


**Abstract—** 
**C**onnecting runtime information to the static properties of the program is a key point in measuring software quality, including testability. Despite a large number of researches on software testability, we observed that the relationship between testability and test adequacy criteria had not been studied, and testability metrics still are far from measuring the actual test effectiveness and effort. We hypothesize that testability has a significant impact on automatic testing tools. Therefore, we propose a new methodology to measure and quantify software testability by exploiting both runtime information and static properties of the source code.
  
**Index Terms:** Software testability, software metrics, code coverage, machine learning.


This page provides supplementary materials for ADAFEST project, including a full list of source code metrics, a full list of case studies, and description of datasets.

## Appendix A: source code metrics
The list of all metrics along with their quality subject, full name, and granularity used in our experimental study. 

|     Subject        	|     Metric abbreviation    	|     Metric full name                                                                                                     	|     Granularity     	|
|--------------------	|----------------------------	|--------------------------------------------------------------------------------------------------------------------------	|---------------------	|
|     Size/Count     	|     CSLOC                  	|     Class line of code                                                                                                   	|     Class           	|
|                    	|     CSNOST                 	|     Class number of statements                                                                                           	|     Class           	|
|                    	|     CSNOSM                 	|     Class number of static   methods                                                                                     	|     Class           	|
|                    	|     CSNOSA                 	|     Class number of static   attributes                                                                                  	|     Class           	|
|                    	|     CSNOIM                 	|     Class number of instance   methods                                                                                   	|     Class           	|
|                    	|     CSNOIA                 	|     Class number of instance   attributes                                                                                	|     Class           	|
|                    	|     CSNOM                  	|     Class number of methods                                                                                              	|     Class           	|
|                    	|     CSNOMNAMM              	|     Class number of not accessor   or mutator methods                                                                    	|     Class           	|
|                    	|     CSNOCON                	|     Class number of constructors                                                                                         	|     Class           	|
|                    	|     CSNOP                  	|     Class number of parameters                                                                                           	|     Class           	|
|                    	|     PKLOC                  	|     Package line of code                                                                                                 	|     Package         	|
|                    	|     PKNOST                 	|     Package number of statements                                                                                         	|     Package         	|
|                    	|     PKNOSM                 	|     Package number of static   methods                                                                                   	|     Package         	|
|                    	|     PKNOSA                 	|     Package number of static   attributes                                                                                	|     Package         	|
|                    	|     PKNOIM                 	|     Package number of instance   methods                                                                                 	|     Package         	|
|                    	|     PKNOIA                 	|     Package number of instance   attributes                                                                              	|     Package         	|
|                    	|     PKNOMNAMM              	|     Package number of not   accessor or mutator methods                                                                  	|     Package         	|
|                    	|     PKNOCS                 	|     Package number of classes                                                                                            	|     Package         	|
|                    	|     PKNOFL                 	|     Package number of files                                                                                              	|     Package         	|
|     Complexity     	|     CSCC                   	|     Class cyclomatic complexity                                                                                          	|     Class           	|
|                    	|     CSNESTING              	|     Class nesting level of   control constructs                                                                          	|     Class           	|
|                    	|     CSPATH                 	|     Class number of unique paths   across a body of code                                                                 	|     Class           	|
|                    	|     CSKNOTS                	|     Measure of overlapping jumps                                                                                         	|     Class           	|
|                    	|     PKCC                   	|     Package cyclomatic complexity                                                                                        	|     Package         	|
|                    	|     PKNESTING              	|     Package nesting level of   control constructs                                                                        	|     Package         	|
|     Cohesion       	|     LOCM                   	|     Lack of Cohesion in Methods                                                                                          	|     Class           	|
|     Coupling       	|     CBO                    	|     Coupling between objects                                                                                             	|     Class           	|
|                    	|     RFC                    	|     Response for a class                                                                                                 	|     Class           	|
|                    	|     FANIN                  	|     Total numbers of inputs a   class functions uses plus the number of unique subprograms calling class   functions.    	|     Class           	|
|                    	|     FANOUT                 	|     Total of functions calls plus   parameters set/modify of class functions                                             	|     Class           	|
|                    	|     DEPENDS                	|     All dependencies of the class                                                                                        	|     Class           	|
|                    	|     DEPENDSBY              	|     Entities depended on by the   class                                                                                  	|     Class           	|
|                    	|     CFNAMM                 	|     Called foreign not accessor   or mutator methods                                                                     	|     Class           	|
|                    	|     ATFD                   	|     Access to foreign data                                                                                               	|     Class           	|
|                    	|     DAC                    	|     Data abstraction coupling                                                                                            	|     Class           	|
|                    	|     NOMCALL                	|     Number of method calls                                                                                               	|     Class           	|
|     Visibility     	|     CSNODM                 	|     Class number of default methods                                                                                      	|     Class           	|
|                    	|     CSNOPM                 	|     Class number of private   methods                                                                                    	|     Class           	|
|                    	|     CSNOPRM                	|     Class number of protected   methods                                                                                  	|     Class           	|
|                    	|     CSNOPLM                	|     Class number of public   methods                                                                                     	|     Class           	|
|                    	|     CSNOAM                 	|     Class number of accessor   methods                                                                                   	|     Class           	|
|                    	|     PKNODM                 	|     Package number of default   methods                                                                                  	|     Package         	|
|                    	|     PKNOPM                 	|     Package number of private   methods                                                                                  	|     Package         	|
|                    	|     PKNOPRM                	|     Package number of protected   methods                                                                                	|     Package         	|
|                    	|     PKNOPLM                	|     Package number of public   methods                                                                                   	|     Package         	|
|                    	|     PKNOAM                 	|     Package number of accessor   methods                                                                                 	|     Package         	|
|     Inheritance    	|     DIT                    	|     Depth of inheritance tree                                                                                            	|     Class           	|
|                    	|     NOC                    	|     Number of children                                                                                                   	|     Class           	|
|                    	|     NOP                    	|     Number of parents                                                                                                    	|     Class           	|
|                    	|     NIM                    	|     Number of inherited methods                                                                                          	|     Class           	|
|                    	|     NMO                    	|     Number of methods overridden                                                                                         	|     Class           	|
|                    	|     NOII                   	|     Number of implemented interfaces                                                                                     	|     Class           	|
|                    	|     PKNOI                  	|     Package number of interfaces                                                                                         	|     Package         	|
|                    	|     PKNOAC                 	|     Package number of abstract   classes                                                                                 	|     Package         	|



## Appendix B: benchmark projects

List of Java projects in SF110 corpus

| Project         | Domain   |   Java files |   Line of codes |
|:----------------|:---------|-------------:|----------------:|
| Jgaap           | -        |           17 |            1009 |
| Netweaver       | -        |          204 |           24380 |
| Squirrel-sql    | -        |         1151 |          122959 |
| Sweethome3d     | -        |          185 |           68704 |
| Vuze            | -        |         3304 |          517207 |
| Freemind        | -        |          472 |           62702 |
| Checkstyle      | -        |          169 |           19575 |
| Weka            | -        |         1031 |          243941 |
| Liferay         | -        |         8345 |         1553460 |
| Pdfsam          | -        |          369 |           35922 |
| Water-simulator | -        |           49 |            6503 |
| Firebird        | -        |          258 |           40005 |
| Imsmart         | -        |           20 |            1039 |
| Dsachat         | -        |           32 |            3861 |
| Jdbacl          | -        |          126 |           18434 |
| Omjstate        | -        |           23 |             593 |
| Beanbin         | -        |           88 |            3788 |
| Templatedetails | -        |            1 |             391 |
| Inspirento      | -        |           36 |            2427 |
| Jsecurity       | -        |          298 |           13134 |
| Jmca            | -        |           25 |           14885 |
| Tullibee        | -        |           20 |            3236 |
| Nekomud         | -        |           10 |             363 |
| Geo-google      | -        |           62 |            6623 |
| Byuic           | -        |           12 |            5874 |
| Jwbf            | -        |           69 |            5848 |
| Saxpath         | -        |           16 |            2620 |
| Jni-inchi       | -        |           24 |            1156 |
| Jipa            | -        |            3 |             392 |
| Gangup          | -        |           95 |           11088 |
| Greencow        | -        |            1 |               6 |
| Apbsmem         | -        |           50 |            4406 |
| A4j             | -        |           45 |            3602 |
| Bpmail          | -        |           37 |            1681 |
| Xisemele        | -        |           56 |            1805 |
| Httpanalyzer    | -        |           19 |            3588 |
| Javaviewcontrol | -        |           17 |            4618 |
| Sbmlreader2     | -        |            6 |             499 |
| Corina          | -        |          349 |           41290 |
| Schemaspy       | -        |           72 |           10008 |
| Petsoar         | -        |           76 |            2255 |
| Javabullboard   | -        |           44 |            8297 |
| Diffi           | -        |           10 |             524 |
| Gaj             | -        |           14 |             320 |
| Glengineer      | -        |           41 |            3124 |
| Follow          | -        |           60 |            4813 |
| Asphodel        | -        |           24 |             691 |
| Lilith          | -        |          295 |           46198 |
| Summa           | -        |          584 |           69341 |
| Lotus           | -        |           54 |            1028 |
| Nutzenportfolio | -        |           84 |            8268 |
| Dvd-homevideo   | -        |            9 |            2913 |
| Resources4j     | -        |           14 |            1242 |
| Diebierse       | -        |           20 |            1888 |
| Rif             | -        |           15 |             953 |
| Biff            | -        |            3 |            2097 |
| Jiprof          | -        |          113 |           13911 |
| Lagoon          | -        |           81 |            9956 |
| Shp2kml         | -        |            4 |             266 |
| Db-everywhere   | -        |          104 |            7125 |
| Lavalamp        | -        |           54 |            1474 |
| Jhandballmoves  | -        |           73 |            5345 |
| Hft-bomberman   | -        |          135 |            8386 |
| Fps370          | -        |            8 |            1506 |
| Mygrid          | -        |           37 |            3317 |
| Templateit      | -        |           19 |            2463 |
| Sugar           | -        |           37 |            3147 |
| Noen            | -        |          408 |           18867 |
| Dom4j           | -        |          173 |           18209 |
| Objectexplorer  | -        |           88 |            6988 |
| Jtailgui        | -        |           44 |            2020 |
| Gsftp           | -        |           17 |            2332 |
| Gae-app-manager | -        |            8 |             411 |
| Biblestudy      | -        |           21 |            2312 |
| Lhamacaw        | -        |          108 |           22772 |
| Jnfe            | -        |           68 |            2096 |
| Echodep         | -        |           81 |           15722 |
| Ext4j           | -        |           45 |            1892 |
| Battlecry       | -        |           11 |            2524 |
| Fim1            | -        |           70 |           10294 |
| Fixsuite        | -        |           25 |            2665 |
| Openhre         | -        |          135 |            8355 |
| Dash-framework  | -        |           22 |             241 |
| Io-project      | -        |           19 |             698 |
| Caloriecount    | -        |          684 |           61547 |
| Twfbplayer      | -        |          104 |            7240 |
| Sfmis           | -        |           19 |            1288 |
| Wheelwebtool    | -        |          113 |           16275 |
| Javathena       | -        |           53 |           10493 |
| Ipcalculator    | -        |           10 |            2684 |
| Xbus            | -        |          203 |           23514 |
| Ifx-framework   | -        |         4027 |          120355 |
| Shop            | -        |           34 |            3894 |
| At-robots2-j    | -        |          231 |            9459 |
| Jaw-br          | -        |           30 |            4851 |
| Jopenchart      | -        |           48 |            3996 |
| Jiggler         | -        |          184 |           20072 |
| Gfarcegestionfa | -        |           50 |            3662 |
| Dcparseargs     | -        |            6 |             204 |
| Classviewer     | -        |            7 |            1467 |
| Jcvi-javacommon | -        |          619 |           45496 |
| Quickserver     | -        |          152 |           16040 |
| Jclo            | -        |            3 |             387 |
| Celwars2009     | -        |           11 |            2876 |
| Heal            | -        |          184 |           22521 |
| Feudalismgame   | -        |           36 |            3515 |
| Trans-locator   | -        |            5 |             357 |
| Newzgrabber     | -        |           39 |            5874 |
| Falselight      | -        |            8 |             372 |


##Appendix C: dataset
The description of DS1 to DS5.


## Release date
The full version of source code will be available as soon as the relevant paper(s) are published.
