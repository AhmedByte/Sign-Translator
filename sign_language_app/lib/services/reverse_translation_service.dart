import 'dart:convert';
import 'package:http/http.dart' as http;

class SignItem {
  final String signId;
  final String labelAr;
  final String labelEn;
  final bool hasGif;
  final List<String> synonyms;

  SignItem({
    required this.signId,
    required this.labelAr,
    required this.labelEn,
    required this.hasGif,
    required this.synonyms,
  });

  factory SignItem.fromJson(Map<String, dynamic> json) {
    return SignItem(
      signId: json['sign_id'] as String,
      labelAr: json['label_ar'] as String,
      labelEn: json['label_en'] as String,
      hasGif: json['has_gif'] as bool? ?? false,
      synonyms: json['synonyms'] != null
          ? List<String>.from(json['synonyms'] as List)
          : [],
    );
  }
}

class WordAnalysis {
  final String word;
  final bool isPerson;
  final bool isMatched;
  final String? bestId;
  final String labelAr;
  final String labelEn;
  final double score;
  final String scorePct;
  bool useSign; // Editable in UI

  WordAnalysis({
    required this.word,
    required this.isPerson,
    required this.isMatched,
    this.bestId,
    required this.labelAr,
    required this.labelEn,
    required this.score,
    required this.scorePct,
    required this.useSign,
  });

  factory WordAnalysis.fromJson(Map<String, dynamic> json) {
    final isMatched = json['is_matched'] as bool? ?? false;
    return WordAnalysis(
      word: json['word'] as String,
      isPerson: json['is_person'] as bool? ?? false,
      isMatched: isMatched,
      bestId: json['best_id'] as String?,
      labelAr: json['label_ar'] as String? ?? '',
      labelEn: json['label_en'] as String? ?? '',
      score: (json['score'] as num? ?? 0.0).toDouble(),
      scorePct: json['score_pct'] as String? ?? '0.0%',
      useSign: isMatched, // Default useSign to true if matched, false otherwise
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'word': word,
      'use_sign': useSign,
      'sign_id': bestId,
    };
  }
}

class ReverseTranslationService {
  static const String baseUrl = 'https://mohamed478-arsl-search.hf.space';

  /// Fetch all available signs in the system dictionary
  static Future<List<SignItem>> getSigns() async {
    final uri = Uri.parse('$baseUrl/api/signs');
    final response = await http.get(uri);

    if (response.statusCode == 200) {
      final List data = jsonDecode(utf8.decode(response.bodyBytes)) as List;
      return data.map((e) => SignItem.fromJson(e as Map<String, dynamic>)).toList();
    } else {
      throw Exception('Failed to load signs: ${response.statusCode}');
    }
  }

  /// Analyze a sentence to detect names and dictionary word matches
  static Future<List<WordAnalysis>> analyzeSentence(String sentence, {double threshold = 0.92}) async {
    final uri = Uri.parse('$baseUrl/api/analyze');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'sentence': sentence,
        'threshold': threshold,
      }),
    );

    if (response.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
      final List words = data['words'] as List;
      return words.map((e) => WordAnalysis.fromJson(e as Map<String, dynamic>)).toList();
    } else {
      throw Exception('Failed to analyze sentence: ${response.statusCode}');
    }
  }

  /// Generate a combined GIF for the analyzed words
  static Future<String> generateSentenceGif(List<WordAnalysis> words, {int fps = 12}) async {
    final uri = Uri.parse('$baseUrl/api/generate');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'words': words.map((w) => w.toJson()).toList(),
        'fps': fps,
      }),
    );

    if (response.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
      if (data['success'] == true && data['gif_url'] != null) {
        // Build the absolute GIF URL
        return '$baseUrl${data['gif_url']}';
      } else {
        throw Exception('Failed to generate sign language GIF.');
      }
    } else {
      throw Exception('Failed to generate sentence GIF: ${response.statusCode}');
    }
  }
}
