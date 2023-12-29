
import language_tool_python

 
my_tool = language_tool_python.LanguageTool('en-US')  
  
#my_text = """So outside oh, but I really want to me tight don't know it feels"""   
my_text="I luv you"   
 
my_matches = my_tool.check(my_text)  
  
print(my_matches)

my_fixes=my_tool.correct(my_text)

print(my_fixes)

from gingerit.gingerit import GingerIt

text = 'I won t do this than'

parser = GingerIt()
fix_sentence=parser.parse(text)
print(fix_sentence['result'])
