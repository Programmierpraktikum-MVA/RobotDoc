from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify
import requests

# huggingface api (reference: https://huggingface.co/d4data/biomedical-ner-all)
API_URL = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def convertString(data):
    output = []
    for d in data:
        score_pct = round(d['score'] * 100, 2)
        line = f"{d['entity_group']}: {d['word']} (score: {score_pct}%, start: {d['start']}, end: {d['end']})"
        output.append(line)
    output.append("")
    return output


def parseString(data):
    # Create a dictionary to hold the arrays
    entity_groups = {}
    # Iterate over each item in the list
    for item in data:
        entity_group = item['entity_group']
        word = item['word']
    # If the entity group already exists, append the word to the existing array
        if entity_group in entity_groups:
            entity_groups[entity_group].append(word)
    # Otherwise, create a new array for the entity group
        else:
            entity_groups[entity_group] = [word]
    return entity_groups
