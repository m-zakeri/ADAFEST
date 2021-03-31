
import sys
import os
from tkinter import *
from statistics import mean




sys.path.insert(0, 'D:/program files/scitools/bin/pc-win64/python')
import understand as und


class cls_main:

    def main(self):
        db = und.open("C:\\Users\\saeed\\Desktop\\alllanguage\\all3\\java.udb")
        countdeaph = 0
        print("sasa")
        obj=cls_get_metrics()
        for ent in db.ents("class"):
            # if str(func.parent().longname()) == "WindowsFormsApplication9.mycal":
                if (ent.longname() == "myhesabdari.saeed"):
                    print(ent.language())
                    print(obj.NOPA(ent))
                    par = ent.ents("define", "Java Variable private Member")
                    for p in par:
                       print("      ",p)
                       countdeaph+=1
        print(countdeaph)


class cls_get_metrics:
    def is_interface(self,classname):
        try:
            if("Interface"in str(classname.kind())  ) :
                return True
            else:
                return False
        except:
            return True


    def get_family_list(self,classname):
        family = list()
        try:
            if(len(classname.refs("Base"))==0):
                family.append(classname)
                return family
        except:
            return family
        family_list=list()
        family_list.append(classname)
        for cla in family_list:
            for f in cla.refs("Base,Derive"):
                if not (f.ent() in family_list)   and  not(len(f.ent().refs("Base"))==0 )     :
                    family_list.append(f.ent())
        return family_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def get_childs(self,classname):
        child_list = list()
        try:
            for f in classname.refs("Derive"):
                 child_list.append(f.ent())
            return child_list
        except:
            return child_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def get_fathers_and_grandfathers(self,classname):
        fathers_list=list()
        try:
            fathers_list.append(classname)
            while(True):
                for f in classname.refs("Base"):
                    parent=f.ent()
                    print("f:",f)
                print("parent        :",parent.name())
                if(parent.name()=="Object" or parent.name()=="page" ):
                    break
                classname=parent
                fathers_list.append(classname)
            return fathers_list
        except:
            return fathers_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def NOPA(self,classname):
        try:
            if(classname.language()=="Java"):
                return len(classname.ents("define","Variable"))
            else:
                return classname.metric(["CountDeclInstanceVariablePublic"])["CountDeclInstanceVariablePublic"]
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def MAXNESTING(self,funcname):
        try:
            if(self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                return funcname.metric(["MaxNesting"])["MaxNesting"]
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def FANOUT(self, funcname):
        try:
            if(self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                return (funcname.metric(["CountOutPut"])["CountOutPut"] )
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def CYCLO (self,funcname):
        try:
            if(self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                return funcname.metric(["Cyclomatic"])["Cyclomatic"]
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # def is_abstract(self,funcname):
    #     if (str(funcname).startswith(("get", "set", "Set", "Get"))   or    funcname.metric(["CountLine"])["CountLine"]<=6):
    #         return True
    #     else:
    #         return False
    def is_abstract(self,ent):
        try:
            if("Abstract" in str(ent.kind())    ):
                return True
            else:
                return False
        except:
            return True
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def NOLV(self,funcname):
        try:
            if(self.is_abstract(funcname) or self.is_interface(funcname.parent())):
                return None
            else:
                # bug
                varlist=funcname.ents("","Variable , Parameter")
                return len(varlist)
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def NOAM(self,class_name):
        try:
            if(self.is_interface(class_name)):
                return None
            else:
                count = 0
                for mth in class_name.ents('Define', 'method'):
                    if (str(mth.name()).startswith(("get", "set", "Set", "Get"))):
                        # print(mth.longname())
                        count += 1
                        # print(mth.longname())
                return count
        except:
            return None

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def NOMNAMM(self,class_name):
        try:
            mth_=class_name.ents("Define","method")
            return  ((len(mth_ ))- self.NOAM(class_name))
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def LOCNAMM(self,class_name):
        try:
            LOC = class_name.metric(["CountLine"])["CountLine"]
            if(LOC==None):
                return None
            else:
                LOCAMM = 0
                for mth in class_name.ents('Define','method'):
                        if (str(mth).startswith(("get","set","Set","Get"))):
                            if (mth.metric(["CountLine"])["CountLine"] != None):
                                 LOCAMM += mth.metric(["CountLine"])["CountLine"]
                return (LOC-LOCAMM)
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def LOC(self,funcname):
        try:
            return funcname.metric(["CountLine"])["CountLine"]
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def WOC(self,class_name):
         try:
             if(self.is_interface(class_name) or self.is_abstract(class_name)):
                 return None
             else:
                count_functionl=0
                metlist=class_name.ents("Define","Public Method")
                for m in metlist:
                    if not ( self.is_abstract(m)):
                        count_functionl+=1

                if(self.NOPA(class_name)==None  or class_name.metric(["CountDeclMethodPublic"])["CountDeclMethodPublic"]==None):
                    return None
                else:
                    total = self.NOPA(class_name) + class_name.metric(["CountDeclMethodPublic"])["CountDeclMethodPublic"]
                    if total == 0:
                        return 0
                    else:
                        return (count_functionl/total)
         except:
             return None

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def WMCNAMM(self,class_name):
        try:
            if(self.is_interface(class_name)):
                return None
            else:
                sum=0
                for mth in class_name.ents('Define','method'):
                    if  not(self.is_accesor_or_mutator(mth)):
                        if(mth.metric(["Cyclomatic"])["Cyclomatic"]!=None):
                            sum += mth.metric(["Cyclomatic"])["Cyclomatic"]
                return sum
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def is_accesor_or_mutator(self,funcname):
        try:
            if (str(funcname).startswith(("get", "set", "Get", "Set"))):
                return True
            else:
                return False
        except:
            return False
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def TCC(self, class_name):
        try:
            if(self.is_abstract(class_name)or self.is_interface(class_name)):
                return None
            else:
                # cal NP
                NDC=0
                methodlist = class_name.ents('Define', 'Public Method')
                method_list_visible=list()
                for mvvisible in methodlist:
                    if self.is_visible(mvvisible):
                        method_list_visible.append(mvvisible)
                for row in range(0,len(method_list_visible)):
                    for col in range(0,len(method_list_visible)):
                        if (row > col):
                            if( self.connectivity(method_list_visible[row],method_list_visible[col])):
                                NDC+=1
                N=len(method_list_visible)
                NP=N*(N-1)/2
                if(NP!=0):
                    return NDC/NP
                else:
                    return 0
        except:
            return None
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def connectivity(self,row,col):
        try:
            if(self.connectivity_directly(row,col)   or self.connectivity_indirectly(row,col) ):
                return True
            else:
                return False
        except:
            return False
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def connectivity_indirectly(self,row,col):
        listrow = set()
        listcol = set()
        try:
            for callrow in row.refs("call"):
                if (str(callrow.ent().name()).startswith(("get",  "Get"))):
                    listrow.add(callrow.ent().longname())
            for callcol in col.refs("call"):
                if (str(callcol.ent().name()).startswith(("get", "Get"))):
                  listcol.add(callcol.ent().longname())
            intersect = [value for value in listrow if value in listcol]
            if (len(intersect) > 0):
                return True
            else:
                return False
        except:
            return False
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def connectivity_directly(self,row,col):
        listrow = set()
        listcol = set()
        try:
            for callrow in row.refs("use"):
                 listrow.add(callrow.ent().longname())
            for callcol in col.refs("use"):
                 listcol.add(callcol.ent().longname())
            intersect = [value for value in listrow if value in listcol]
            if(len(intersect)>0):
                return True
            else:
                return False
        except:
            return False
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def is_visible(self,funcname):
        try:
            flag=False
            # """all parameter not  only use or declare """
            par = funcname.ents("", "Parameter")
            for p in par:
                if(str(p.type())=="EventArgs"):
                    flag=True
                    break
            if not(     str(funcname.kind())=="Public Constructor")      or      not(flag)      or      not(str(funcname.kind())=="Private Method"):
                 return True
            else:
                return False
        except:
            return False
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def CDISP(self,method_name):
        try:
            if (self.is_abstract(method_name) or self.is_interface(method_name.parent())):
                return None
            else:
                cint = self.CINT(method_name)

                if cint==0:
                    return 0
                elif(cint==None):
                    return None
                else:
                    return self.FANOUT_OUR(method_name)/cint
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def FANOUT_OUR(self,methodname):
        try:
            if (self.is_abstract(methodname) or self.is_interface(methodname.parent())):
                return None
            else:
                called_classes_set=set()
                for call in methodname.refs("call"):
                    if (call.ent().library() != "Standard"):
                        called_classes_set.add(  call.ent().parent() )
                return len(called_classes_set)
        except:
            return None

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def ATLD(self,db,method_name):
        try:
            if (  not(self.is_abstract(method_name)) or  not(self.is_interface(method_name.parent()))   ):
                # cal directly access
                count=0
                system_att_=db.ents('field')
                access_att_=self.give_access_use(method_name)
                for att_ in access_att_:
                    if att_ in system_att_:
                        if (str(att_.kind()) in ["Unknown Variable", "Unknown Class"]):
                            continue
                        if(att_.library()!="Standard"):
                            count+=1
                # cal indirectly access
                calls=self.give_ALL_sys_and_lib_method_that_th_measyred_method_calls( method_name)
                for call in calls:
                    if (str(call).startswith(("get", "Get"))):
                        usevariable=call.refs("use")
                        if( len(usevariable)>0):
                            flag=True
                            for us in usevariable:
                                if(us.ent().library()=="Standard"):
                                    flag=False
                            if(flag):
                                count += 1
                return count
            else:
                return None
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def give_ALL_sys_and_lib_method_that_th_measyred_method_calls(self,funcname):

        methodlist=set()
        try:
            for refmethod in funcname.refs("call"):
                methodlist.add(refmethod.ent())
            return methodlist
        except:
            return methodlist
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def get_Namespace(self,entity):
        while (str(entity.parent().kind()) != "Unresolved Namespace" or str(entity.parent().kind()) != "Namespace"):
            entity = entity.parent()
            if (str(entity.parent().kind()) == "Unresolved Namespace" or str(   entity.parent().kind()) == "Namespace"):
                break
        return entity.parent()
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def ATFD_method(self, db, method_name):

        if not (self.is_accesor_or_mutator(method_name)):
            count = 0
            system_att_ = db.ents('Variable')
            access_att_ = self.give_access_use(method_name)
            for att_ in access_att_:
                if not (att_ in system_att_):
                    count += 1
            # cal indirectly access
            calls = self.give_Methods_that_the_measured_method_calls(method_name)
            for call in calls:
                if (self.is_accesor_or_mutator(call)):
                    get_att_ = self.give_access_use(call)
                    if not (get_att_ in system_att_):
                        count += 1
            return count
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # checked ok successfully ( ckeck field of librabry ex:math.PI isincluded or not ? at :if(var_.parent() not in family_list and var_.library()!="Standard"   ):
    def ATFD_CLASS(self,class_namee):
        try:
            count = 0
            family_list=self.get_family_list(class_namee)
            varibleset = set()
            methodlist = class_namee.ents('Define', 'Public Method')
            for methodname in methodlist:
                    # directly
                    if( not(self.is_abstract(methodname)) and methodname.name()!=class_namee.name()):
                        method_accessvariable = self.give_access_use(methodname)
                        for var_ in method_accessvariable:
                            if("Field" in str(var_.kind()) ):
                                if(var_.parent() not in family_list and var_.library()!="Standard"   ):
                                    varibleset.add(var_)
                                    count+=1
                        # indirectly
                        method_called_list = self.give_Methods_that_the_measured_method_calls(methodname)
                        # print(method_called_list)
                        for m in method_called_list:
                            # print(m.parent())
                            if(m.parent() not in family_list  and  str(m).startswith(("get","Get"))):
                                varibleset.add(m)
                                count+=1
            return len(varibleset)
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def max_depth(self,method_name):
        # at least two
        if len(self.give_Methods_that_the_measured_method_calls(method_name)) ==0:
            return 0
        #if self.give_Methods_that_the_measured_method_calls(method_name).count() ==1:
        #    return 1
        return 1+ max([self.Mamcl(node) for node in self.give_Methods_that_the_measured_method_calls(method_name)])
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def total_nods(self,method_name):
        if self.give_Methods_that_the_measured_method_calls(method_name).count() ==0:
            return 0
        if self.give_Methods_that_the_measured_method_calls(method_name).count() ==1:
            return 1
        return 1+ sum([self.Mamcl(node) for node in self.give_Methods_that_the_measured_method_calls(method_name)])
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def NMCS(self, method_name):
        try:
            count=0
            for mth in  self.give_Methods_that_the_measured_method_calls(method_name):
                if str(mth).startswith("get","set","Get","Set"):
                    count+=1
            return count
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def MaMCL(self,method_name):
        try:
            max_dep=self.max_depth(method_name)
            if max_dep>=2:
                return max_dep
            else:
                return 0
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def MeMCL(self,method_name):
        try:
            total=self.total_nods(method_name)
            nmcs=self.NMCS(method_name)
            if nmcs==0:
                return 0
            else:
                return round(total/nmcs)
        except:
            return None

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # def CC(self,funcname):
    #     refset = set()
    #     reflist = list()
    #     for callint in funcname.refs("callby"):
    #         refset.add(callint.ent().parent())
    #         #reflist.append(callint.ent().parent())
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def CM(self,funcname):
        try:
            if(self.is_private(funcname)):
                return None
            else:
                refset = set()
                for callint in funcname.refs("callby"):
                    refset.add(callint.ent())
                return len(refset)
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def is_private (slef,funcname):
        try:
            if(str(funcname.kind())=="Private Method"):
                return True
            else:
                return False
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def CINT(self,method_name):
        try:
            if(self.is_abstract(method_name)  or self.is_interface(method_name.parent())):
                return None
            else:
                count=0
                family_list=self.get_family_list(method_name.parent())
                for mth in method_name.refs("call"):
                       if(  mth.ent().parent() in family_list):
                           count+=1
                return count
        except:
            return None

    def LAA(self, input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                result = 0
                listtotal = set()
                listsameclass = set()
                for vr in input_method.refs('Use', 'Variable'):
                    listtotal.add(vr.ent())
                    if (vr.ent().parent() == input_method.parent()):
                        listsameclass.add(vr.ent())
                for meth in input_method.refs('call', 'method'):
                    if meth.ent().kindname() != "Public Constructor":
                        if (self.is_accesor_or_mutator(meth.ent())):
                            listtotal.add(meth.ent())
                        if (self.is_accesor_or_mutator(meth.ent()) and meth.ent().parent() == input_method.parent()):
                            listsameclass.add(meth.ent())
                if (len(listtotal) != 0):
                    result = len(listsameclass) / len(listtotal)
                return result
        except:
            return 0

    def FDP(self, input_method):
        try:
            if (self.is_abstract(input_method)):
                return 0
            else:
                listmethod = set()
                for meth in input_method.refs('Use', 'Variable'):
                    if (meth.ent().parent() != input_method.parent()):
                        listmethod.add(meth.ent().parent())
                return len(listmethod)
        except:
            return 0

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def give_access_use(self,funcname):
        # create a list and return it:Includes all the variables(fields) that a method uses
        access_field_list = set()
        try:
            for fi in funcname.refs("use"):
                access_field_list.add(fi.ent())
            return access_field_list
        except:
            return access_field_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def give_access_use_for_class(self,classname):
        # create a list and return it:Includes all the variables(fields) that a method uses
        access_field_list = list()
        try:
            for fi in classname.refs("use"):
                access_field_list.append(fi.ent())
            return access_field_list
        except:
            return access_field_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def give_Methods_that_the_measured_method_calls(self,funcname):
        call_methods_list = set()
        try:
            for fi in funcname.refs("call"):
                if( fi.ent().library()!="Standard"):
                    call_methods_list.add(fi.ent())
            return call_methods_list
        except:
            return call_methods_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def give_cc(self,db,funcname):
        try:
            if(self.is_private(funcname)):
                return None
            else:
                refset = set()
                for callint in funcname.refs("callby"):
                    refset.add(callint.ent().parent())
                return len((refset))
        except:
            return None
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def give_Methods_that_the_measured_class_calls(self,classname):
        # create a list and return it:Includes all Methods entity(also cunstructor method ) that the measured method calls
        call_methods_list = list()
        try:
            for fi in classname.refs("call"):
                # if namespace == method namespace
                if (fi.ent().parent().parent() == classname.parent()  ):
                    call_methods_list.append(fi.ent())
            return call_methods_list
        except:
            return call_methods_list
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def returt_result(self,db):
        self.get_metrics(db)
        return [self.class_metrics, self.method_metrics]
        # return a list consist of classes and methods and thier metrics value

obj=cls_main()

