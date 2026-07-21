# 배포 체크리스트

- [x] 전체 pytest 통과
- [x] `streamlit run app.py` 정상 실행
- [x] 로컬 절대경로 없음
- [x] 코드·문서에 비밀정보 없음
- [x] `requirements.txt` 호환 범위 지정
- [x] Python 버전 지정
- [x] 실제 EPRX·JEPX 데이터 Git 제외
- [x] 샘플 데이터 표시 명확
- [x] EPRX 파일 없음 처리
- [x] JEPX 파일 없음 처리
- [ ] 모바일 또는 좁은 화면 수동 확인
- [x] 주요 그래프 자동 실행 확인
- [x] widget key 충돌 없음
- [x] README 배포 안내 최신화
- [ ] GitHub 저장소 생성 및 업로드

## 배포 전 수동 확인

1. 저장소에 실제 시장 CSV·Excel과 `.streamlit/secrets.toml`이 없는지 확인합니다.
2. Streamlit Community Cloud의 Advanced settings에서 Python 3.14를 선택합니다.
3. 엔트리포인트를 `app.py`로 지정합니다.
4. 실제 데이터가 필요하면 공개 저장소가 아닌 승인된 저장소·스토리지 정책을 먼저 확인합니다.
5. 배포 로그에서 의존성 설치와 데이터 폴더 부재 안내가 정상인지 확인합니다.
