import os
import shutil
import sys

import numpy as np
from nltk import Tree
# from nltk.draw import TreeWidget
# from nltk.draw.util import CanvasFrame

from . import common
from . import relation_set
from . import utils_dis_thiago
from . import utils_rs3


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class Document:
    def __init__(self, dpath):  # ? parse=parse, raw=raw
        self.path = dpath
        self.datatype = None
        self.tree = None
        self.tokendict = None  # Token dict: id token in the document -> token form
        self.eduIds = []
        self.edudict = None  # EDU dict: id EDU -> list of id tokens
        self.outbasename = os.path.basename(self.path)  # Name of the output file, can be modified for the RST DT
        self.statistics = {}  # statistics for one document

    def read(self):
        raise NotImplementedError

    def writeTree(self, outpath, outExt):
        '''
        Write the bracketed tree into a file
        Remove the original extension, keep only .outExt as extension
        '''
        fileout = os.path.join(outpath,
                               self.outbasename.replace('.out', '').replace('.txt.lisp', '').replace(
                                   '.' + self.datatype, '')) + outExt
        with open(fileout, 'w') as fout:
            fout.write(self.tree.__str__().strip())

    def drawTree(self, outpath, ext, outExt, docno=-1):
        '''Draw RST tree into a file'''
        pass

    def mapRelation(self, mappingRel):
        if self.tree == None:
            return
        if os.path.isfile(mappingRel):
            sys.exit("Mapping RS3 from file not implemented yet")
        else:
            if mappingRel == 'mapping':  # Default general mapping
                common.performMapping(self.tree, relation_set.mapping)
            elif mappingRel == 'basque_labels':
                common.performMapping(self.tree, relation_set.basque_labels)
            elif mappingRel == 'brazilianCst_labels':
                common.performMapping(self.tree, relation_set.brazilianCst_labels)
            elif mappingRel == 'brazilianSum_labels':
                common.performMapping(self.tree, relation_set.brazilianSum_labels)
            elif mappingRel == 'germanPcc_labels':
                common.performMapping(self.tree, relation_set.germanPcc_labels)
            elif mappingRel == 'spanish_labels':
                common.performMapping(self.tree, relation_set.spanish_labels)
            elif mappingRel == 'rstdt_mapping18':
                common.performMapping(self.tree, relation_set.rstdt_mapping18)
            elif mappingRel == 'dutch_labels':
                common.performMapping(self.tree, relation_set.dutch_labels)
            elif mappingRel == 'brazilianTCC_labels':
                common.performMapping(self.tree, relation_set.brazilianTCC_labels)
            else:
                print("Unknown mapping: " + mappingRel)


class Rs3Document(Document):
    '''
    Class for a document encoded in rs3 format.
    - XML format
    - the relation list in the header gives the nuclearity of the relations
    - EDU id are not always continuous: EDU are renamed
    - For some corpora/languages, the binarization using right branching is not enough,
    a more general strategy is used
    - An EDU file is created
    '''

    def __init__(self, dpath):
        Document.__init__(self, dpath)
        self.datatype = "rs3" if dpath[-3:] == 'rs3' else 'rs4'
        self.nuclearity_relations = {}

    def read(self):
        '''
        Create a binarized (NLTK) Tree, self.tree, from the rs3 file
        Fill self.tokendict and self.edudict
        '''
        doc_root, rs3_xml_tree = utils_rs3.parse_xml(self.path)
        # Retrieve the relations in the header (used to find multinuc rel)
        self.nuclearity_relations = utils_rs3.get_relations_type(rs3_xml_tree)
        # Get info for each node
        eduList, groupList, root = utils_rs3.readRS3Annotation(doc_root)
        # Build nodes, rename DU, tree=SpanNode instance
        try:
            tree = utils_rs3.buildNodes(eduList, groupList, root, self.nuclearity_relations)
        except Exception as e:
            print(f'No Root node in the file:', self.path)
            raise Exception

        # Can t be retrieved from the tree for now, some EDU have children
        eduIds = [e["id"] for e in eduList]
        # Order span list for each node
        utils_rs3.orderSpanList(tree, eduIds)
        # Clean the tree: deal with DU with only one child + same unit cases
        utils_rs3.cleanTree(tree, eduIds, self.nuclearity_relations, self)
        # Retrieve info about the text of the EDUs
        self.tokendict, self.edudict = utils_rs3.retrieveEdu(tree, eduIds)
        # non_bin_tree = tree
        # Binarize the tree
        utils_rs3.binarizeTreeGeneral(tree, self, nucRelations=self.nuclearity_relations)
        tree = common.backprop(tree, self)  # Backprop info
        self.tree = Tree.fromstring(common.parse(tree))  # Build an nltk tree
        validTree = common.checkTree(self.tree, self)
        if not validTree:
            self.tree = None

    def writeEdu(self, outpath):
        utils_rs3.writeEdus(self, outpath)


# ----------------------------------------------------------------------------------
class DisDocument(Document):
    def __init__(self, dpath, epath):
        Document.__init__(self, dpath)
        self.datatype = "dis"
        self.eduPath = epath

    def read(self):  # , eduFiles
        basename = os.path.basename(self.path)
        for e in ['.out', '.dis', '.txt', '.edus']:
            basename = basename.replace(e, '')
        if basename in utils_dis_thiago.file_mapping:  # Modify the name of some specific files in the RST DT
            self.outbasename = utils_dis_thiago.file_mapping[basename]
        tree, self.eduIds = utils_dis_thiago.buildTree(open(self.path).read())  # Build RST Tree
        tree = utils_dis_thiago.binarizeTreeRight(tree)  # Binarize it
        # doc = utils_dis_thiago.readEduDoc(self.eduPath, self)  # Retrieve info on EDUs
        tree = common.backprop(tree, self)
        str_tree = common.parse(tree)  # Get nltk tree
        self.tree = Tree.fromstring(str_tree)

    def writeEdu(self, outpath):
        # copy the EDU file, possibly rename it using the file mapping
        if self.outbasename != os.path.basename(self.path.split('.')[0]):
            shutil.copy(self.eduPath, os.path.join(outpath,
                                                   self.outbasename.replace('.out', '').replace('.dis', '') + '.edus'))
        else:
            shutil.copy(self.eduPath.replace('.out', '').replace('.dis', ''), outpath)


# ----------------------------------------------------------------------------------
class ThiagoDocument(Document):
    def __init__(self, dpath):
        Document.__init__(self, dpath)
        self.datatype = "thiago"
        self.eduPath = None

    def read(self):
        tree, self.eduIds, allnodes, self.edudict = utils_dis_thiago.buildTreeThiago(
            open(self.path, encoding="windows-1252").read())
        tree = utils_dis_thiago.bTree(allnodes, self.path)
        tree = utils_dis_thiago.binarizeTreeRightThiago(tree)
        tree = common.backprop(tree, self)  # Backprop info
        self.tree = Tree.fromstring(common.parse(tree))

    def writeEdu(self, outpath):
        common.writeEdusFile(self, ".txt.lisp.thiago", outpath)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class SpanNode:
    """
    RST tree node (from DPLP, by Yangfeng Ji)
    """

    def __init__(self, prop):
        """
        Initialization of SpanNode
        :type text: string
        :param text: text of this span
        """
        self.text, self.relation = None, None  # Text of this span / Discourse relation
        self.eduspan, self.nucspan = None, None  # EDU span / Nucleus span (begin, end) index id EDU
        self.nucedu = None  # Nucleus single EDU (itself id for an EDU)s
        self.prop = prop  # Property: Nucleus/Satellite/Roots
        self.lnode, self.rnode = None, None  # Children nodes (for binary RST tree only)
        self.pnode = None  # Parent node
        self.nodelist = []  # Node list (for general RST tree only)
        self.form = None  # Relation form: NN, NS, SN
        self.eduCovered = []  # Id of the EDUS covered by a CDU (CHLOE Added)
        self._id = None  # Id (int) of a DU, only from rs3 files (CHLOE Added)

    def __str__(self):
        return self._info() + "\n" + "\n".join("\t" + n._info() for n in self.nodelist)

    def _info(self):
        return "eduspan: " + str(self.eduspan)


# ----------------------------------------------------------------------------------
def associate_tree_edus(treeFiles, eduFiles):
    ''' Retrieve the EDU file associated to a tree for the dis format '''
    documents = []
    for treePath in treeFiles:
        basename = os.path.basename(treePath)
        for e in ['.out', '.dis', '.txt', '.edus']:
            basename = basename.replace(e, '')
        eduPath = utils_dis_thiago.findFile(eduFiles, basename)  # Retrieve EDUs file
        if eduPath == None:
            sys.exit("Edus file not found: " + basename)
        documents.append(DisDocument(treePath, eduPath))
    return documents


def getFiles(tbpath, ext):
    files = []
    for p, dirs, _files in os.walk(tbpath):
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in _files:
            if not file.startswith('.') and file.endswith(ext):
                files.append(os.path.join(p, file))
    return files
