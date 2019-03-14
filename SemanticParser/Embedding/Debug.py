# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function : test


import os

from SemanticParser.Embedding.TrainCharacterEmbedding import TrainCharacterEmbedding

if __name__ == '__main__':
    root_path = os.path.abspath('/') + 'Users\\10928\\Desktop' + '\\IntelligentService'
    faq_path = root_path + '\\KnowledgeMemory' + '\\FAQ' + '\\FAQ.csv'
    json_faq_path = root_path + '\\KnowledgeMemory' + '\\FAQ' + '\\FAQ.json'
    print(faq_path)

    # debug trainsentencesembedding
    # emb_path = root_path + '\\KnowledgeMemory' + '\\Embedding' + '\\SentenceEmbedding'
    # print(emb_path)
    # solution = GetSentenceEmbedding(filepath=faq_path, title='问题', save_path=emb_path)
    # content = solution.get_text(faq_path, title='问题')
    # content = solution.df2list(content)
    # sen_vec = solution.get_sentence_embedding(content)
    # result = solution.save_sentence_embedding(save_path=emb_path, save_name='SentencesEmbedding.json', sentences_embedding=sen_vec)

    # debug csv2json
    # c2j = Csv2Json(read_path=faq_path, write_path=json_faq_path)
    # dic = c2j.read_csv()
    # c2j.write_json(dic)

    # debug traincharacterembedding
    emb_path = root_path + '\\KnowledgeMemory' + '\\Embedding' + '\\CharacterEmbedding'
    print(emb_path)
    json_question_character_embedding_path = '\\CharactersEmbedding.json'
    # json_answer_character_embedding_path = '\\answer_characters_embedding.json'
    train = TrainCharacterEmbedding(json_faq_path, emb_path)
    dataframe = train.get_text()
    all_characters = train.split_character(dataframe)
    characters_embedding = train.train_character_embedding(all_characters)
    train.save_character_embedding(characters_embedding, emb_path, json_question_character_embedding_path)
    # train.save_character_embedding(answer_embedding, emb_path, json_answer_character_embedding_path)
