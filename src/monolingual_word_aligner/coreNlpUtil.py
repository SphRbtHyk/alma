# flake8: noqa
from typing import Any
from stanza.models.common.doc import Document
from monolingual_word_aligner.config import *


##############################################################################################################################
def parseText(sentences, nlp):

    parseResult: Document = nlp.parse(sentences)

    parse_result_dict = parseResult.to_dict()
    nb_sentences = len(parse_result_dict)
    #print(f"nb sentences: {nb_sentences}")

    # NOTE: !!!!!!!!!!!!!!!! This fix only works for single sentence parsing !!!!!!!!!!!!!!!!!!!
    #if len(parseResult['sentences']) == 1:
    if nb_sentences == 1:
        return parseResult

    raise NotImplementedError("The code for multiple sentences has not been updated for use with stanza package.")

    wordOffset = 0

    #for i in range(len(parseResult['sentences'])):
    for i in range(nb_sentences):

        if i>0:
            for j in range(len(parseResult['sentences'][i]['dependencies'])):

                for k in range(1,3):
                    tokens = parseResult['sentences'][i]['dependencies'][j][k].split('-')
                    if tokens[0] == 'ROOT':
                        newWordIndex = 0
                    else:
                        if not tokens[len(tokens)-1].isdigit(): # forced to do this because of entries like u"lost-8'" in parseResult
                            continue
                        newWordIndex = int(tokens[len(tokens)-1])+wordOffset
                    if len(tokens) == 2:
                        parseResult['sentences'][i]['dependencies'][j][k] = tokens[0]+ '-' + str(newWordIndex)
                    else:
                        w = ''
                        for l in range(len(tokens)-1):
                            w += tokens[l]
                            if l<len(tokens)-2:
                                w += '-'
                        parseResult['sentences'][i]['dependencies'][j][k] = w + '-' + str(newWordIndex)

        wordOffset +=  len(parseResult['sentences'][i]['words'])


    # merge information of all sentences into one
    for i in range(1,len(parseResult['sentences'])):
        parseResult['sentences'][0]['text'] += ' ' + parseResult['sentences'][i]['text']
        for jtem in parseResult['sentences'][i]['dependencies']:
            parseResult['sentences'][0]['dependencies'].append(jtem)
        for jtem in parseResult['sentences'][i]['words']:
            parseResult['sentences'][0]['words'].append(jtem)

    # remove all but the first entry
    parseResult['sentences'] = parseResult['sentences'][0:1]

    return parseResult
##############################################################################################################################



##############################################################################################################################
def nerWordAnnotator(parseResult: list[list[dict[str, Any]]]):

    res = []

    #wordIndex = 1
    #for i in range(len(parseResult['sentences'][0]['words'])):
    #    tag = [[parseResult['sentences'][0]['words'][i][1]['CharacterOffsetBegin'],
    #            parseResult['sentences'][0]['words'][i][1]['CharacterOffsetEnd']],
    #            wordIndex,
    #            parseResult['sentences'][0]['words'][i][0],
    #            parseResult['sentences'][0]['words'][i][1]['NamedEntityTag']]
    #    wordIndex += 1
    #
    #    if tag[3] != 'O':
    #        res.append(tag)

    for word_idx in range(len(parseResult[0])):
        named_entity_value = parseResult[0][word_idx]['ner']
        if named_entity_value != 'O':
            tag = [[parseResult[0][word_idx]['start_char'], 
                    parseResult[0][word_idx]['end_char']],
                   word_idx + 1,
                   parseResult[0][word_idx]['text'], # NOTE: Not sure this is the right object
                   named_entity_value]
            res.append(tag)


    return res
##############################################################################################################################


##############################################################################################################################
def ner(parseResult):

    nerWordAnnotations = nerWordAnnotator(parseResult)

    namedEntities = []
    currentNE = []
    currentCharacterOffsets = []
    currentWordOffsets = []

    for i in range(len(nerWordAnnotations)):

               
        if i == 0:
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if len(nerWordAnnotations) == 1:
                namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i-1][3]])
                break
            continue

        if nerWordAnnotations[i][3] == nerWordAnnotations[i-1][3] and nerWordAnnotations[i][1] == nerWordAnnotations[i-1][1]+1:
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if i == len(nerWordAnnotations)-1:
                namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i][3]])
        else:
            namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i-1][3]])
            currentNE = [nerWordAnnotations[i][2]]
            currentCharacterOffsets = []
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets = []
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if i == len(nerWordAnnotations)-1:
                namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i][3]])

    #print namedEntities  

    return namedEntities    
##############################################################################################################################


##############################################################################################################################
def posTag(parseResult):

    res = []

    #wordIndex = 1
    #for i in range(len(parseResult['sentences'][0]['words'])):
    #    tag = [[parseResult['sentences'][0]['words'][i][1]['CharacterOffsetBegin'],
    #            parseResult['sentences'][0]['words'][i][1]['CharacterOffsetEnd']],
    #            wordIndex,
    #            parseResult['sentences'][0]['words'][i][0],
    #            parseResult['sentences'][0]['words'][i][1]['PartOfSpeech']]
    #    wordIndex += 1
    #    res.append(tag)

    for word_idx in range(len(parseResult[0])):
        tag = [[parseResult[0][word_idx]['start_char'], 
                parseResult[0][word_idx]['end_char']],
               word_idx + 1,
               parseResult[0][word_idx]['text'], # NOTE: Not sure this is the right object
               parseResult[0][word_idx]['upos']]
        res.append(tag)

    return res
##############################################################################################################################




##############################################################################################################################
def lemmatize(parseResult: list[list[dict[str, Any]]]):

    res = []

    #wordIndex = 1
    #for i in range(len(parseResult['sentences'][0]['words'])):
    #    tag = [[parseResult['sentences'][0]['words'][i][1]['CharacterOffsetBegin'],
    #            parseResult['sentences'][0]['words'][i][1]['CharacterOffsetEnd']],
    #            wordIndex,
    #            parseResult['sentences'][0]['words'][i][0],
    #            parseResult['sentences'][0]['words'][i][1]['Lemma']]
    #    wordIndex += 1
    #    res.append(tag)

    for word_idx in range(len(parseResult[0])):
        tag = [[parseResult[0][word_idx]['start_char'], 
                parseResult[0][word_idx]['end_char']],
               word_idx + 1,
               parseResult[0][word_idx]['text'], # NOTE: Not sure this is the right object
               parseResult[0][word_idx]['lemma']]
        res.append(tag)

    return res
##############################################################################################################################





##############################################################################################################################
def dependencyParseAndPutOffsets(parseResult):
# returns dependency parse of the sentence whhere each item is of the form (rel, left{charStartOffset, charEndOffset, wordNumber}, right{charStartOffset, charEndOffset, wordNumber})

    #dParse = parseResult['sentences'][0]['dependencies']
    #words = parseResult['sentences'][0]['words']

    #for item in dParse:
        #print item

    #result = []

    #for item in dParse:
    #    newItem = []

        # copy 'rel'
    #    newItem.append(item[0])

        # construct and append entry for 'left'
    #    left = item[1][0:item[1].rindex("-")]
    #    wordNumber = item[1][item[1].rindex("-")+1:]
    #    if wordNumber.isdigit() == False:
    #        continue
    #    left += '{' + words[int(wordNumber)-1][1]['CharacterOffsetBegin'] + ' ' + words[int(wordNumber)-1][1]['CharacterOffsetEnd'] + ' ' + wordNumber + '}'
    #    newItem.append(left)

        # construct and append entry for 'right'
    #    right = item[2][0:item[2].rindex("-")]
    #    wordNumber = item[2][item[2].rindex("-")+1:]
    #    if wordNumber.isdigit() == False:
    #        continue
    #    right += '{' + words[int(wordNumber)-1][1]['CharacterOffsetBegin'] + ' ' + words[int(wordNumber)-1][1]['CharacterOffsetEnd'] + ' ' + wordNumber  + '}'
    #    newItem.append(right)

    #    result.append(newItem)
    result = []

    for i, word in enumerate(parseResult[0]):
        if (i <= 0) or (i >= len(parseResult[0]) - 1):
            continue
        left =  str(parseResult[0][i - 1]["text"]) + '{' + str(parseResult[0][i - 1]["start_char"]) + ' ' + str(parseResult[0][i - 1]["end_char"]) + ' ' + str(parseResult[0][i - 1]["id"]) + '}'
        right =  str(parseResult[0][i + 1]["text"]) + '{' + str(parseResult[0][i + 1]["start_char"]) + ' ' + str(parseResult[0][i + 1]["end_char"]) + ' ' + str(parseResult[0][i + 1]["id"]) + '}'
        result.append([word["deprel"], left, right])

    return result
##############################################################################################################################



##############################################################################################################################
def findParents(dependencyParse, wordIndex, word):
# word index assumed to be starting at 1
# the third parameter is needed because of the collapsed representation of the dependencies...

    wordsWithIndices = ((int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0]) for item in dependencyParse)
    wordsWithIndices = list(set(wordsWithIndices))
    wordsWithIndices = sorted(wordsWithIndices, key=lambda item: item[0])
    #print wordsWithIndices

    wordIndexPresentInTheList = False
    for item in wordsWithIndices:
        if item[0] == wordIndex:
            wordIndexPresentInTheList = True
            break

    parentsWithRelation = []

    if wordIndexPresentInTheList:
        for item in dependencyParse:
            currentIndex = int(item[2].split('{')[1].split('}')[0].split(' ')[2])
            if currentIndex == wordIndex:
                parentsWithRelation.append([int(item[1].split('{')[1].split('}')[0].split(' ')[2]), item[1].split('{')[0], item[0]])
    else:
        # find the closest following word index which is in the list
        nextIndex = 0
        for i in range(len(wordsWithIndices)):
            if wordsWithIndices[i][0] > wordIndex:
                nextIndex = wordsWithIndices[i][0]
                break
        if nextIndex == 0:
            return [] #?
        for i in range(len(dependencyParse)):
            if int(dependencyParse[i][2].split('{')[1].split('}')[0].split(' ')[2]) == nextIndex:
                   pos = i
                   break
        for i in range(pos, len(dependencyParse)):
            if '_' in dependencyParse[i][0] and word in dependencyParse[i][0]:
                parent = [int(dependencyParse[i][1].split('{')[1].split('}')[0].split(' ')[2]), dependencyParse[i][1].split('{')[0], dependencyParse[i][0]]
                parentsWithRelation.append(parent)
                break
        
    return parentsWithRelation
##############################################################################################################################




##############################################################################################################################
def findChildren(dependencyParse, wordIndex, word):
# word index assumed to be starting at 1
# the third parameter is needed because of the collapsed representation of the dependencies...

    wordsWithIndices = ((int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0]) for item in dependencyParse)
    wordsWithIndices = list(set(wordsWithIndices))
    wordsWithIndices = sorted(wordsWithIndices, key=lambda item: item[0])

    wordIndexPresentInTheList = False
    for item in wordsWithIndices:
        if item[0] == wordIndex:
            wordIndexPresentInTheList = True
            break

    childrenWithRelation = []

    if wordIndexPresentInTheList:
        #print True
        for item in dependencyParse:
            currentIndex = int(item[1].split('{')[1].split('}')[0].split(' ')[2])
            if currentIndex == wordIndex:
                childrenWithRelation.append([int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0], item[0]])
    else:
        # find the closest following word index which is in the list
        nextIndex = 0
        for i in range(len(wordsWithIndices)):
            if wordsWithIndices[i][0] > wordIndex:
                nextIndex = wordsWithIndices[i][0]
                break

        if nextIndex == 0:
            return []
        for i in range(len(dependencyParse)):
            if int(dependencyParse[i][2].split('{')[1].split('}')[0].split(' ')[2]) == nextIndex:
                   pos = i
                   break
        for i in range(pos, len(dependencyParse)):
            if '_' in dependencyParse[i][0] and word in dependencyParse[i][0]:
                child = [int(dependencyParse[i][2].split('{')[1].split('}')[0].split(' ')[2]), dependencyParse[i][2].split('{')[0], dependencyParse[i][0]]
                childrenWithRelation.append(child)
                break
        
    return childrenWithRelation
##############################################################################################################################


# nlp = StanfordNLP()
# nlp = StanzaParser()