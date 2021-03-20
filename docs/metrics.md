
# ADAFEST metrics

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

