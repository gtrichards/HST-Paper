"""Build the page from template + data, then run its script under
JavaScriptCore with a stand-in DOM, expanding the given rows through the
page's own click handler.  Usage: run_page_test.py WORKDIR row row ..."""
import json, re, subprocess, sys, os
S, rows = sys.argv[1], sys.argv[2:]; H = os.path.dirname(os.path.abspath(__file__))
t = open(S + "/template.html").read()
out = t.replace("__DATA__", open(S + "/zcheck_data.json").read().replace("</", "<\\/")).replace("__SPEC__", open(S + "/zcheck_spectra.json").read().replace("</", "<\\/"))
open(S + "/redshift_check.html", "w").write(out)
js = re.findall(r"<script>(.*?)</script>", out, re.S)[0]
tail = ("var res=[]; [%s].forEach(function(row){ var b=ctxCalls; handlers['doc:click']({target:{closest:function(s){ return s==='.exp'?{dataset:{row:String(row)}}:null; }}}); res.push('row '+row+': '+(ctxCalls-b)+' drawing calls'); });"
        "'OK. table renders '+LOG.filter(function(x){return x.indexOf('tb html')===0;}).length+'; '+res.join('; ');") % ",".join(rows)
full = open(H + "/stub_head.js").read() + "\nvar DATA_TXT=" + json.dumps(open(S + "/zcheck_data.json").read()) + ";\nvar SPEC_TXT=" + json.dumps(open(S + "/zcheck_spectra.json").read()) + ";\nvar SPEC_OBJ=JSON.parse(SPEC_TXT);\n" + js + "\n" + tail
open(S + "/run_test.js", "w").write("try{ eval(" + json.dumps(full) + "); }catch(e){ 'RUNTIME ERROR: '+e }")
open(S + "/page.js", "w").write(js)
for f in ("page.js",):
    print(subprocess.run(["osascript", "-l", "JavaScript", H + "/check.js"], env=dict(os.environ, JSFILE=S + "/" + f), capture_output=True, text=True).stdout.strip())
print(subprocess.run(["osascript", "-l", "JavaScript", S + "/run_test.js"], capture_output=True, text=True).stdout.strip())
