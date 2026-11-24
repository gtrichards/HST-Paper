ObjC.import('Foundation');
var p = $.NSProcessInfo.processInfo.environment.objectForKey('JSFILE').js;
var src = $.NSString.stringWithContentsOfFileEncodingError(p, $.NSUTF8StringEncoding, null).js;
try { new Function(src); 'SYNTAX OK (' + src.length + ' chars)'; } catch (e) { 'SYNTAX ERROR: ' + e; }
