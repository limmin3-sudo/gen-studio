from genstudio import router
from genstudio.registry import Registry

REG = Registry.load()


def pick(prompt, **kw):
    return router.choose(REG, prompt, **kw)


def test_default_style_is_realistic():
    d = pick("a woman reading in a cafe")
    assert d.style == "realistic"
    assert d.model.id == "cyberrealistic-zit"
    assert (d.width, d.height) == (1024, 1024)


def test_semi_wins_over_realistic_substring():
    # "반실사" contains "실사"; it must still route to semi-real
    d = pick("반실사 느낌의 여자 캐릭터")
    assert d.style == "semi"
    assert d.model.id == "realism-illustrious"


def test_realistic_keyword():
    assert pick("인물 사진, 자연광").style == "realistic"
    assert pick("raw photo of a sailor, 85mm").style == "realistic"


def test_explicit_style_and_model():
    assert pick("anything", style="semi").model.family.id == "illustrious"
    d = pick("anything", model_id="semireal-mm")
    assert d.model.id == "semireal-mm" and d.style == "semi"


def test_aspect_detection():
    assert pick("세로 전신 사진").aspect == "portrait"
    assert pick("wide landscape at dusk").aspect == "landscape"
    d = pick("x", aspect="tall")
    assert (d.width, d.height) == (768, 1344)


def test_custom_size_snaps_to_16():
    d = pick("x", width=1010, height=1500)
    assert (d.width, d.height) == (1008, 1504)


def test_split_negative():
    assert router.split_negative("a cat --no hat, text") == ("a cat", "hat, text")
    assert router.split_negative("a cat") == ("a cat", "")


def test_edit_uses_source_model():
    m, why = router.model_for_edit(REG, "semireal-mm", None)
    assert m.id == "semireal-mm"
    m, why = router.model_for_edit(REG, None, None)
    assert m.style == "realistic"


def test_text_and_poster_requests_go_to_qwen():
    for p in ["카페 간판에 '봄날' 글자", '네온사인으로 "OPEN" 이라고 쓴 사진', "movie poster for a space opera", "회사 로고 디자인"]:
        d = pick(p)
        assert d.style == "text" and d.model.id == "qwen-image-21-turbo", p
    # plain portrait requests are unaffected
    assert pick("창가에 앉은 여자 실사").model.id == "cyberrealistic-zit"


def test_keyword_picks_wan():
    d = pick("극실사 인물 사진, 창가")
    assert d.model.id == "wan22-14b" and d.style == "realistic"
    assert pick("극실사", style="semi").model.family.id == "illustrious"   # explicit style wins
