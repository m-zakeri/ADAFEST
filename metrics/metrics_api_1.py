"""

"""
import sys
from collections import Counter

sys.path.insert(0, "D:/program files/scitools/bin/pc-win64/python")
import understand

from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, db=None,
                 project_name: str = None, ):
        self.db = db
        self.project_name = project_name

    @abstractmethod
    def compute_metric(self, metric_name: str = None):
        pass


class PackageMetric(Metric):
    def __init__(self, db=None,
                 project_name: str = None,
                 package_name: str = None, ):
        super(PackageMetric, self).__init__(db=db, project_name=project_name)
        self.package_name = package_name

    def compute_metric(self, metric_name: str = None):
        pass


class ClassMetric(PackageMetric):
    def __init__(self, db=None,
                 project_name: str = None,
                 package_name: str = None,
                 class_name: str = None,
                 method_name: str = None):
        super(ClassMetric, self).__init__(db=db, project_name=project_name, package_name=package_name)
        self.class_name = class_name



    def compute_metric(self, metric_name: str = None):
        if metric_name == 'MinCyclomatic':
            pass



class MethodMetric(ClassMetric):
    def __init__(self, db=None,
                 project_name: str = None,
                 package_name: str = None,
                 class_name: str = None,
                 method_name: str = None):
        super(MethodMetric, self).__init__(db=db,
                                           project_name=project_name,
                                           package_name=package_name,
                                           class_name=class_name)
        self.method_name = method_name

    def compute_metric(self, metric_name: str = None):
        pass


class UnderstandUtility(object):
    """
    https://scitools.com/documents/manuals/python/understand.html
    https://scitools.com/documents/manuals/perl/#ada_entity_kinds
    """

    @classmethod
    def get_db_tokens(cls, db, look_up_string=".\.cc|.\.h"):
        token_types = ['Newline', 'Whitespace', 'Indent', 'Dedent', 'Comment']
        files = db.lookup(look_up_string, 'file')
        # files = db.lookup('.', 'file')
        print('files:', len(files))
        number_of_all_tokens = 0
        number_of_identifiers = 0
        number_of_error = 0
        token_type_list = list()

        for file in files:
            print('-' * 50, file)
        # input()
        for file in files:
            print('-' * 50, file)
            # if file.name().find('.pb.') != -1 or file.name() == 'logging.h':
            #     continue
            try:
                for lexeme in file.lexer():
                    print(lexeme.text(), ': ', lexeme.token())
                    # if lexeme.ent():
                    #     print('@', lexeme.ent() )
                    if lexeme.token() == 'Identifier':
                        number_of_identifiers += 1
                    if lexeme.token() not in token_types:
                        number_of_all_tokens += 1
                    token_type_list.append(lexeme.token())
            except:
                print('ERROR!')
                number_of_error += 1
            # input()
        print('All tokens:', number_of_all_tokens)
        print('All identifiers:', number_of_identifiers)
        print('identifier ratio to all tokens:', 100 * number_of_identifiers / number_of_all_tokens, '%')
        print('error', number_of_error)
        counter = Counter(token_type_list)
        print('All token types:', counter)

    # -------------------------------------------
    # Getting Types list: Class (three method), Abstract Class, Interface, Enum, Type
    @classmethod
    def get_class_names(cls, db):
        class_name_list = list()
        entities = db.ents('Class ~Unresolved')
        for class_ in sorted(entities, key=UnderstandUtility.sort_key):
            print(class_.name())
            class_name_list.append(class_.longname())
        # print('PJNOCN', len(class_name_list))
        return class_name_list

    # Java specific method
    @classmethod
    def get_project_classes_longnames_java(cls, db):
        class_name_list = list()
        entities = db.ents('Java Class ~Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')
        # entities = db.ents('Type')
        for class_ in sorted(entities, key=UnderstandUtility.sort_key):
            # print(class_.name())
            class_name_list.append(class_.longname())
        # print('PJNOCN', len(class_name_list))
        return class_name_list

    @classmethod
    def get_project_classes_java(cls, db):
        entities = db.ents('Java Class ~Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')
        # entities = db.ents('Type')
        # print('PJNOC', len(entities))
        return entities

    @classmethod
    def get_project_abstract_classes_java(cls, db):
        entities = db.ents('Java Abstract Class ~Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')
        # print('PJNOAC', len(entities))
        return entities

    @classmethod
    def get_project_interfaces_java(cls, db):
        entities = db.ents('Java Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')
        # print('PJNOI', len(entities))
        return entities

    @classmethod
    def get_project_enums_java(cls, db):
        entities = db.ents('Java Java Enum ~Unknown ~Unresolved ~Jar ~Library')
        # print('PJNOENU', len(entities))
        return entities

    @classmethod
    def get_project_types_java(cls, db):
        entities = db.ents('Type')
        # entities = db.ents('Java Class')
        # print('PJNOT', len(entities))
        return entities

    # -------------------------------------------
    # Getting Types individually with their name
    @classmethod
    def get_class_entity_by_name(cls, db, class_name):
        # https://docs.python.org/3/library/exceptions.html#exception-hierarchy
        # Find relevant 'class' entity
        entity_list = list()

        # entities = db.ents('Type')  ## Use this for evo-suite SF110 already measured class
        entities = db.ents('Java Class ~Enum ~Unknown ~Unresolved ~Jar ~Library')
        if entities is not None:
            for entity_ in entities:
                if entity_.longname() == class_name:
                    entity_list.append(entity_)
                    # print('Class entity:', entity_)
                    # print('Class entity kind:', entity_.kind())
        if len(entity_list) == 0:
            # raise UserWarning('Java class with name {0} is not found in project'.format(class_name))
            return None
        if len(entity_list) > 1:
            raise ValueError('There is more than one Java class with name {0} in the project'.format(class_name))
        else:
            return entity_list[0]

    @classmethod
    def get_base_metric(cls, db, class_name):
        class_entity = UnderstandUtility.get_entity_by_name(db=db, class_name=class_name)


    @classmethod
    def get_method_of_class_java(cls, db, class_name):
        method_list = list()
        # entities = db.ents('function, method Member ~Unresolved')
        entities = db.ents('Java Method')
        # print(class_name)
        for method_ in sorted(entities, key=UnderstandUtility.sort_key):
            # print(method_)
            # print(method_.parent().longname())
            if method_.parent() is None:
                continue
            if str(method_.parent().longname()) == class_name:
                # print('\tname:', method_.name(), '\tkind:', method_.kind().name(), '\ttype:', method_.type())
                method_list.append(method_)
        # print('len method list', len(method_list))
        # print(method_list)
        return method_list

    @classmethod
    def get_method_of_class_java2(cls, db, class_name=None, class_entity=None):
        """
        Both methods 'get_method_of_class_java' and 'get_method_of_class_java2' works correctly.
        :param db:
        :param class_name:
        :param class_entity:
        :return:
        """
        if class_entity is None:
            class_entity = cls.get_class_entity_by_name(db=db, class_name=class_name)
        method_list = class_entity.ents('Define', 'Java Method ~Unknown ~Unresolved ~Jar ~Library')
        # print('len method list', len(method_list))
        # print(method_list)
        return method_list

    @classmethod
    def get_constructor_of_class_java(cls, db, class_name=None, class_entity=None):
        """
        :param db:
        :param class_name:
        :param class_entity:
        :return:
        """
        if class_entity is None:
            class_entity = cls.get_class_entity_by_name(db=db, class_name=class_name)
        constructor_list = class_entity.ents('Define', 'Java Method Constructor')
        # print('len constructor list', len(constructor_list))
        # print(constructor_list)
        return constructor_list

    @classmethod
    def get_method_name_of_class(cls, db, class_name):
        method_name_list = list()
        entities = db.ents('function, method Member ~Unresolved')
        # print(class_name)
        for method_ in sorted(entities, key=UnderstandUtility.sort_key):
            if str(method_.parent()) == class_name:
                # print('\tname:', method_.name(), '\tkind:', method_.kind().name(), '\ttype:', method_.type())
                method_name_list.append(method_.name())
        return method_name_list

    @classmethod
    def get_attribute_of_class(cls, db, class_name):
        attribute_name_list = list()
        # entities = db.ents('Object member ~Unresolved')  # For my C# project works well but not for Java projects
        entities = db.ents('Variable')
        print(class_name)
        for attr_ in sorted(entities, key=UnderstandUtility.sort_key):
            if str(attr_.parent()) == class_name:
                # print('\t', attr_.name(), attr_.kind().name())
                # print('\tname:', attr_.name(), '\tkind:', attr_.kind().name(), '\ttype:', attr_.type())
                attribute_name_list.append(attr_.name())
        return attribute_name_list

    @classmethod
    def get_class_attributes_java(cls, db, class_name=None, class_entity=None) -> list:
        if class_entity is None:
            class_entity = UnderstandUtility.get_class_entity_by_name(db=db, class_name=class_name)
        class_attributes_list = list()
        for java_var in class_entity.ents('Define', 'Java Variable'):
            # print(java_var.longname())
            # print(java_var.kind())
            # print('TYPE::', java_var.type())
            # print(java_var.library())
            # print('-------------')
            class_attributes_list.append(java_var)

        return class_attributes_list

    @classmethod
    def get_data_abstraction_coupling(cls, db, class_name=None, class_entity=None) -> int:
        java_primitive_types = ['byte', 'short', 'int', 'long', 'float', 'double',
                                'boolean', 'char',
                                'String'
                                ]
        attrs = UnderstandUtility.get_class_attributes_java(db, class_name=class_name, class_entity=class_entity)
        dac = 0
        for attr in attrs:
            if attr.type() not in java_primitive_types:
                dac += 1
        return dac

    @classmethod
    def get_number_of_class_in_file_java(cls, db, class_name=None, class_entity=None):
        """
        :param db:
        :param class_name:
        :param class_entity:
        :return:
        """
        if class_entity is None:
            class_entity = cls.get_class_entity_by_name(db=db, class_name=class_name)

        number_of_class_in_class_file = class_entity.parent().ents('Define', 'Java Class ~Unknown ~Unresolved ~Jar ~Library')
        # print('number_of_class_in_class_file:', len(number_of_class_in_class_file))
        return number_of_class_in_class_file

    @classmethod
    def get_package_of_given_class(cls, db, class_name):
        # Find package: strategy 2: Dominated strategy
        class_name_list = class_name.split('.')[:-1]
        package_name = '.'.join(class_name_list)
        # print('package_name string', package_name)
        package_list = db.lookup(package_name + '$', 'Package')
        if package_list is None:
            return None
        if len(package_list) == 0:  # if len != 1 return None!
            return None
        package = package_list[0]
        print(package.longname())
        return package

    @classmethod
    def get_package_of_given_class_2(cls, db, class_name):
        class_entity = UnderstandUtility.get_class_entity_by_name(db, class_name)
        # print(class_entity.parent())
        # print('class_name', class_entity.longname())
        package_list = class_entity.ents('Containin', 'Java Package')
        # print('package_name', package_list)
        return package_list[0]

    @classmethod
    def get_package_clasess_java(cls, db=None, package_entity=None):
        # This method has a bug! (dataset version 0.3.0, 0.4.0)
        # Bug is now solved.
        # classes = package_entity.ents('Contain', 'class')
        # interfaces = package_entity.ents('Contain', 'interface')
        all_types = package_entity.ents('Contain', 'Java Type ~Unknown ~Unresolved ~Jar ~Library')
        # print('number of classes', len(classes))
        # print('number of interface', len(interfaces))
        # print('number of all types', len(all_types))
        # for class_entity in classes:
        #     print(class_entity.longname())
        # print('-'*75)
        # for interface_entity in interfaces:
        #     print(interface_entity.longname())

        # for type_entity in all_types:
        #     print(type_entity.longname(),
        #           type_entity.kind(),
        #           type_entity.metric(['CountLineCode'])['CountLineCode'],
        #           type_entity.metric(['CountLineCodeDecl'])['CountLineCodeDecl'],
        #           type_entity.metric(['CountLineCodeExe'])['CountLineCodeExe'],
        #           type_entity.metric(['AvgLineCode'])['AvgLineCode'],)

        return all_types

    @classmethod
    def get_package_classes_by_accessor_method_java(cls, db=None, package_entity=None, accessor_method=''):
        # classes = package_entity.ents('Contain', 'class')
        # interfaces = package_entity.ents('Contain', 'interface')
        all_types = package_entity.ents('Contain', "Java Abstract Enum Type Default Member" + accessor_method)
        # print('number of classes', len(classes))
        # print('number of interface', len(interfaces))
        # print('Number of all interfaces', len(all_types))
        for type_entity in all_types:
            print(type_entity.longname(),
                  type_entity.kind(),
                  )
        return all_types

    @classmethod
    def get_package_interfaces_java(cls, db=None, package_entity=None):
        # classes = package_entity.ents('Contain', 'class')
        # interfaces = package_entity.ents('Contain', 'interface')
        all_acs = package_entity.ents('Contain', 'Java Interface ~Unknown ~Unresolved')
        # print('number of classes', len(classes))
        # print('number of interface', len(interfaces))
        # print('Number of package interfaces:', len(all_acs))
        return all_acs

    @classmethod
    def get_package_abstract_class_java(cls, db=None, package_entity=None):
        # classes = package_entity.ents('Contain', 'class')
        # interfaces = package_entity.ents('Contain', 'interface')
        abstract_classes = package_entity.ents('Contain', 'Java Abstract Class ~Unknown ~Unresolved')
        # print('number of classes', len(classes))
        # print('number of interface', len(interfaces))
        # print('Number of package abstract class', len(abstract_classes))
        return abstract_classes

    @classmethod
    def get_project_files_java(cls, db):
        files = db.ents('Java File ~Jar')
        print('Number of files', len(files))
        # for file_entity in files:
        #     print(file_entity.longname(),
        #           file_entity.kind(),
                  # file_entity.metric(['CountLineCode'])['CountLineCode'],
                  # file_entity.metric(['CountLineCodeDecl'])['CountLineCodeDecl'],
                  # file_entity.metric(['CountLineCodeExe'])['CountLineCodeExe'],
                  # file_entity.metric(['AvgLineCode'])['AvgLineCode'],
                  # file_entity.metric(['CountStmtDecl'])['CountStmtDecl'],
                  # file_entity.metric(['CountStmtDecl'])['CountStmtDecl'],
                  # file_entity.metric(['SumCyclomatic'])['SumCyclomatic'],
                  # )


        return files

    @classmethod
    def get_local_variables(cls, db, function_name):
        local_var_name_list = list()
        entities = db.ents(' Object Local ~Unresolved')
        print(function_name)
        for attr_ in sorted(entities, key=UnderstandUtility.sort_key):
            if str(attr_.parent()) == function_name:
                # print('\t', attr_.name(), attr_.kind().name())
                # print('\tname:', attr_.name(), '\tkind:', attr_.kind().name(), '\ttype:', attr_.type())
                local_var_name_list.append(attr_.name())
        return local_var_name_list

    @classmethod
    def funcWithParameter(cls, db):
        # Test understandability
        ents = db.ents("function, method, procedure")
        for func in sorted(ents, key=UnderstandUtility.sort_key):
            # If the file is from the Ada Standard library, skip to the next
            if func.library() != "Standard":
                print(func.longname(), ' - ', func.name(), '==>', func.parent(), " --> ", func.parameters(), sep="",
                      end="\n")
                # func.draw('Control Flow', 'cfg.png')

    @classmethod
    def draw_cfg_for_class_java(cls, db, class_name=None, class_entity=None):
        """
        :param db:
        :param class_name:
        :param class_entity:
        :return:
        """
        if class_entity is None:
            class_entity = cls.get_class_entity_by_name(db=db, class_name=class_name)

        # class_entity.draw('Declaration', 'Declaration_graph.jpg')
        class_entity.draw('Control Flow Graph', 'CFG_graph.jpg')

    # -------------------------------------------
    @classmethod
    def ATFD(cls, db, class_entity=None, class_name=None):
        java_primitive_types = ['byte', 'short', 'int', 'long', 'float', 'double',
                                'boolean', 'char',
                                'String'
                                ]
        if class_entity is None:
            class_entity = UnderstandUtility.get_class_entity_by_name(db, class_name=class_name)

        methods = class_entity.ents('Define', 'Java Method')
        all_fd_list = set()
        for method_ in methods:
            # print(method_.simplename(), '|', method_.parent(), '|', method_.kind())

            foreign_data = method_.ents('Java Define', 'Java Variable')
            # foreign_data = method_.ents('Java Use', 'Java Variable')
            # foreign_data = class_entity.ents('Modify', 'Java Variable')
            # print('Number of ATFD:', len(set(foreign_data)))

            # all_fd_list.extend(set(foreign_data))
            for fd in foreign_data:
                # print(fd.longname(), '| ', fd.parent(), '| ', fd.kind(), '| ', fd.type())
                if fd.type() not in java_primitive_types:
                    all_fd_list.add(fd.type())
            # print('-'*75)
        # print('all FD:', len(all_fd_list))
        return len(all_fd_list)

    @classmethod
    def NOII(cls, db):
        noii = 0
        interfaces = UnderstandUtility.get_project_interfaces_java(db)
        for interface in interfaces:
            usin = interface.ents('Useby', 'Java Class ~Jar')
            if usin is not None and len(usin) > 0:
                noii += 1
            # print(interface.longname(), '| ', interface.kind(), '|', interface.parent(), '|', usin)
            # print('-'*75)
        # print('Number of implemented interface: ', noii)
        return noii

    @classmethod
    def number_of_method_call(cls, db=None, class_entity=None, class_name=None):
        if class_entity is None:
            class_entity = UnderstandUtility.get_class_entity_by_name(db=db, class_name=class_name)
        method_calls = class_entity.ents('Call', )
        # print('method_calls:', len(method_calls))
        # print(method_calls)
        return len(method_calls)

    @classmethod
    def sort_key(cls, ent):
        return str.lower(ent.longname())

    @classmethod
    def get_entity_kind(cls, db, class_name):
        entity = db.lookup(class_name + '$', 'Type')
        return entity[0].kindname()



if __name__ == '__main__':
    # Sample Database path
    # path = '../input_source/DemoProjectForSTART/DemoProjectForSTART.udb'
    path = '../input_source/apollo5-master/understand/understand_analysis.udb'
    path = '../testability/sf110_without_test/110_firebird.udb'
    path = '../testability/sf110_without_test/107_weka.udb'
    # path = '../testability/sf110_without_test/101_netweaver.udb'
    db = understand.open(path)

    # Test for dataset version 0.3.0

    # all_attr = UnderstandUtility.get_class_attributes_java(db=db, class_name='org.firebirdsql.jdbc.FBDataSource')
    # print(all_attr)
    # dac = UnderstandUtility.get_data_abstraction_coupling(db=db,
    #                                                       class_name=r'org.firebirdsql.jdbc.FBDataSource')
    # print(dac)

    # UnderstandUtility.ATFD(db=db, class_name=r'org.firebirdsql.jdbc.AbstractDriver')
    # UnderstandUtility.NOII(db=db)
    # UnderstandUtility.number_of_method_call(db=db, class_name=r'org.firebirdsql.jdbc.AbstractDriver')

    print('-'*75)

    # pk = UnderstandUtility.get_package_of_given_class(db=db,
    #                                              class_name=r'org.firebirdsql.jdbc.FBDataSource')
    # UnderstandUtility.get_package_of_given_class_2(db=db,
                                                   # class_name=r'org.firebirdsql.jdbc.FBDataSource'
                                                   # class_name=r'org.firebirdsql.jdbc.parser.StatementParser'
                                                   # )

    # UnderstandUtility.get_package_interfaces_java(package_entity=pk)
    # UnderstandUtility.get_package_abstract_class_java(package_entity=pk)
    # UnderstandUtility.get_package_classes_by_accessor_method_java(package_entity=pk, accessor_method='Default')
    # UnderstandUtility.get_project_files_java(db=db)

    # Test for dataset version 0.5.0
    # l2 = UnderstandUtility.get_project_classes_longnames_java(db=db)
    #
    # l3 = UnderstandUtility.get_project_classes_java(db=db)
    # l4 = UnderstandUtility.get_project_interfaces_java(db=db)
    # l5 = UnderstandUtility.get_project_abstract_classes_java(db=db)
    # l6 = UnderstandUtility.get_project_enums_java(db=db)
    #
    # l7 = UnderstandUtility.get_project_types_java(db=db)
    # l8 = UnderstandUtility.get_constructor_of_class_java(db=db, class_name='weka.gui.graphvisualizer.DotParser')

    method_list = UnderstandUtility.get_method_of_class_java2(db=db, class_name='weka.gui.graphvisualizer.DotParser')
    # for i, method in enumerate(method_list):
    #     print(i+1, method.longname())
    #     for j, m in enumerate(method.metrics()):
    #         print('\t', j+1, method.metric([m]))

    # UnderstandUtility.get_number_of_class_in_file_java(db=db, class_name='weka.gui.graphvisualizer.DotParser')
    # UnderstandUtility.draw_cfg_for_class_java(db=db, class_name='weka.gui.graphvisualizer.DotParser')
    # UnderstandUtility.get_class_entity_by_name(db=db, class_name='com.sap.netweaver.porta.core.ApplicationStatus')
